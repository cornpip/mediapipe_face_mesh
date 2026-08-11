#include "mediapipe_face.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#else
#include "tensorflow/lite/c/c_api.h"
#endif
#else
#include "tensorflow/lite/c/c_api.h"
#endif

#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite_runtime.h"

#if defined(__ANDROID__)
#include <android/log.h>
#define MP_DETECT_LOG_TAG "MediapipeFaceDetector"
#define MP_DETECT_LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, MP_DETECT_LOG_TAG, __VA_ARGS__)
#define MP_DETECT_LOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, MP_DETECT_LOG_TAG, __VA_ARGS__)
#else
#define MP_DETECT_LOGI(...) std::fprintf(stdout, "[INFO] " __VA_ARGS__)
#define MP_DETECT_LOGE(...) std::fprintf(stderr, "[ERROR] " __VA_ARGS__)
#endif

namespace {

struct RectInPixels {
  float center_x = 0.0f;
  float center_y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float rotation = 0.0f;
};

struct TensorTransform {
  float center_x = 0.0f;
  float center_y = 0.0f;
  float roi_width = 0.0f;
  float roi_height = 0.0f;
  float rotation = 0.0f;
  float scale = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
  float scaled_width = 0.0f;
  float scaled_height = 0.0f;
};

struct ProjectionMatrix {
  float m00 = 1.0f;
  float m01 = 0.0f;
  float m02 = 0.0f;
  float m10 = 0.0f;
  float m11 = 1.0f;
  float m12 = 0.0f;
};

struct RgbPixel {
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
};

struct Anchor {
  float x_center = 0.0f;
  float y_center = 0.0f;
  float width = 1.0f;
  float height = 1.0f;
};

struct DecodedDetection {
  float left = 0.0f;
  float top = 0.0f;
  float right = 0.0f;
  float bottom = 0.0f;
  float score = 0.0f;
  std::array<float, 12> keypoints{};
};

float Clamp(float value, float min_value, float max_value) {
  return std::max(min_value, std::min(max_value, value));
}

int ClampInt(int value, int min_value, int max_value) {
  return std::max(min_value, std::min(max_value, value));
}

float NormalizeAngle(float radians) {
  constexpr float kPi = 3.14159265358979323846f;
  constexpr float kTwoPi = kPi * 2.0f;
  float angle = radians;
  while (angle > kPi) {
    angle -= kTwoPi;
  }
  while (angle < -kPi) {
    angle += kTwoPi;
  }
  return angle;
}

float Sigmoid(float value) {
  const float clamped = Clamp(value, -100.0f, 100.0f);
  return 1.0f / (1.0f + std::exp(-clamped));
}

float IntersectionOverUnion(const DecodedDetection& a,
                            const DecodedDetection& b) {
  const float left = std::max(a.left, b.left);
  const float top = std::max(a.top, b.top);
  const float right = std::min(a.right, b.right);
  const float bottom = std::min(a.bottom, b.bottom);
  const float width = std::max(0.0f, right - left);
  const float height = std::max(0.0f, bottom - top);
  const float intersection = width * height;
  const float area_a =
      std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top);
  const float area_b =
      std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top);
  const float denominator = area_a + area_b - intersection;
  if (denominator <= 1e-6f) {
    return 0.0f;
  }
  return intersection / denominator;
}

DecodedDetection WeightedMergeDetections(
    const std::vector<const DecodedDetection*>& group) {
  DecodedDetection merged;
  if (group.empty()) {
    return merged;
  }

  float total_weight = 0.0f;
  float max_score = 0.0f;
  for (const DecodedDetection* detection : group) {
    const float weight = std::max(detection->score, 1e-6f);
    total_weight += weight;
    max_score = std::max(max_score, detection->score);
    merged.left += detection->left * weight;
    merged.top += detection->top * weight;
    merged.right += detection->right * weight;
    merged.bottom += detection->bottom * weight;
    for (size_t i = 0; i < merged.keypoints.size(); ++i) {
      merged.keypoints[i] += detection->keypoints[i] * weight;
    }
  }

  if (total_weight > 0.0f) {
    const float inv_total_weight = 1.0f / total_weight;
    merged.left *= inv_total_weight;
    merged.top *= inv_total_weight;
    merged.right *= inv_total_weight;
    merged.bottom *= inv_total_weight;
    for (float& value : merged.keypoints) {
      value *= inv_total_weight;
    }
  }
  merged.score = max_score;
  return merged;
}

MpNormalizedRect ToNormalizedRect(const RectInPixels& rect,
                                  int image_width,
                                  int image_height) {
  MpNormalizedRect normalized{};
  if (image_width <= 0 || image_height <= 0) {
    return normalized;
  }
  normalized.x_center = rect.center_x / image_width;
  normalized.y_center = rect.center_y / image_height;
  normalized.width = rect.width / image_width;
  normalized.height = rect.height / image_height;
  normalized.rotation = rect.rotation;
  return normalized;
}

std::vector<Anchor> GenerateShortRangeAnchors() {
  constexpr int kInputSize = 128;
  constexpr float kMinScale = 0.1484375f;
  constexpr float kMaxScale = 0.75f;
  constexpr int kNumLayers = 4;
  constexpr int kStrides[kNumLayers] = {8, 16, 16, 16};

  std::vector<Anchor> anchors;
  anchors.reserve(896);

  for (int layer = 0; layer < kNumLayers;) {
    std::vector<float> scales;
    int last_same_stride_layer = layer;
    while (last_same_stride_layer < kNumLayers &&
           kStrides[last_same_stride_layer] == kStrides[layer]) {
      const float scale =
          kMinScale + (kMaxScale - kMinScale) * last_same_stride_layer /
                          static_cast<float>(kNumLayers - 1);
      scales.push_back(scale);
      const float next_scale =
          (last_same_stride_layer == kNumLayers - 1)
              ? 1.0f
              : (kMinScale + (kMaxScale - kMinScale) *
                                 (last_same_stride_layer + 1) /
                                 static_cast<float>(kNumLayers - 1));
      scales.push_back(std::sqrt(scale * next_scale));
      ++last_same_stride_layer;
    }

    const int stride = kStrides[layer];
    const int feature_map_size = static_cast<int>(
        std::ceil(static_cast<float>(kInputSize) / stride));
    for (int y = 0; y < feature_map_size; ++y) {
      for (int x = 0; x < feature_map_size; ++x) {
        const float anchor_x =
            (static_cast<float>(x) + 0.5f) / feature_map_size;
        const float anchor_y =
            (static_cast<float>(y) + 0.5f) / feature_map_size;
        for (size_t scale_index = 0; scale_index < scales.size();
             ++scale_index) {
          Anchor anchor;
          anchor.x_center = anchor_x;
          anchor.y_center = anchor_y;
          anchor.width = 1.0f;
          anchor.height = 1.0f;
          anchors.push_back(anchor);
        }
      }
    }
    layer = last_same_stride_layer;
  }

  return anchors;
}

std::vector<Anchor> GenerateFullRangeAnchors() {
  constexpr int kInputSize = 192;
  constexpr int kStride = 4;
  constexpr int kFeatureMapSize = kInputSize / kStride;

  std::vector<Anchor> anchors;
  anchors.reserve(kFeatureMapSize * kFeatureMapSize);
  for (int y = 0; y < kFeatureMapSize; ++y) {
    for (int x = 0; x < kFeatureMapSize; ++x) {
      Anchor anchor;
      anchor.x_center =
          (static_cast<float>(x) + 0.5f) / kFeatureMapSize;
      anchor.y_center =
          (static_cast<float>(y) + 0.5f) / kFeatureMapSize;
      anchor.width = 1.0f;
      anchor.height = 1.0f;
      anchors.push_back(anchor);
    }
  }
  return anchors;
}

class FaceDetectorContext {
 public:
  FaceDetectorContext() = default;
  ~FaceDetectorContext() { Shutdown(); }

  bool Initialize(const std::string& model_path,
                  const MpFaceDetectorCreateOptions* options) {
    threads_ = 2;
    if (options && options->threads > 0) {
      threads_ = options->threads;
    }
    min_detection_confidence_ =
        (options && options->min_detection_confidence > 0.0f)
            ? options->min_detection_confidence
            : 0.5f;
    min_suppression_threshold_ =
        (options && options->min_suppression_threshold > 0.0f)
            ? options->min_suppression_threshold
            : 0.3f;
    max_results_ = (options && options->max_results > 0) ? options->max_results
                                                         : 1;

    const char* runtime_path =
        (options && options->tflite_library_path)
            ? options->tflite_library_path
            : nullptr;
    if (!runtime_.Load(runtime_path)) {
      SetError("Failed to load TensorFlow Lite runtime: " + runtime_.error());
      return false;
    }

    model_.reset(runtime_.ModelCreateFromFile(model_path.c_str()));
    if (!model_) {
      SetError("Unable to load model file: " + model_path);
      return false;
    }

    options_.reset(runtime_.InterpreterOptionsCreate());
    if (!options_) {
      SetError("Failed to allocate interpreter options.");
      return false;
    }
    runtime_.InterpreterOptionsSetThreads(options_.get(), threads_);

    const MpDelegateType delegate_choice =
        options ? static_cast<MpDelegateType>(options->delegate)
                : MP_DELEGATE_CPU;
    const bool allow_delegate_fallback =
        !options || options->disable_delegate_fallback == 0;
    active_delegate_ = MP_DELEGATE_CPU;
    auto AttachDelegate = [&](TfLiteDelegate* created,
                              TfLiteDelegateDeleter::DeleteFn deleter,
                              const char* name,
                              MpDelegateType delegate_type) {
      if (!created) {
        return false;
      }
      delegate_.get_deleter().deleter = deleter;
      delegate_.reset(created);
      runtime_.InterpreterOptionsAddDelegate(
          options_.get(),
          reinterpret_cast<TfLiteOpaqueDelegate*>(delegate_.get()));
      active_delegate_ = delegate_type;
      MP_DETECT_LOGI("%s delegate enabled.\n", name);
      return true;
    };

    switch (delegate_choice) {
      case MP_DELEGATE_XNNPACK: {
        if (!runtime_.InterpreterOptionsAddDelegate ||
            !runtime_.XnnpackDelegateOptionsDefault ||
            !runtime_.XnnpackDelegateCreate ||
            !runtime_.XnnpackDelegateDelete) {
          if (!allow_delegate_fallback) {
            SetError("XNNPACK delegate is unavailable for face detector and "
                     "delegate fallback is disabled.");
            return false;
          }
          MP_DETECT_LOGI(
              "XNNPACK delegate requested but unavailable in runtime.\n");
          break;
        }
        TfLiteXNNPackDelegateOptions delegate_options =
            runtime_.XnnpackDelegateOptionsDefault();
        delegate_options.num_threads = threads_;
        TfLiteDelegate* created =
            runtime_.XnnpackDelegateCreate(&delegate_options);
        if (!AttachDelegate(created, runtime_.XnnpackDelegateDelete,
                            "XNNPACK", MP_DELEGATE_XNNPACK)) {
          if (!allow_delegate_fallback) {
            SetError("Failed to create XNNPACK delegate for face detector "
                     "because delegate fallback is disabled.");
            return false;
          }
          MP_DETECT_LOGE(
              "Failed to create XNNPACK delegate. Falling back to CPU.\n");
        }
        break;
      }
      case MP_DELEGATE_GPU_V2: {
        if (!runtime_.InterpreterOptionsAddDelegate ||
            !runtime_.GpuDelegateV2OptionsDefault ||
            !runtime_.GpuDelegateV2Create || !runtime_.GpuDelegateV2Delete) {
          if (!allow_delegate_fallback) {
            SetError("GPU delegate (V2) is unavailable for face detector and "
                     "delegate fallback is disabled.");
            return false;
          }
          MP_DETECT_LOGI(
              "GPU delegate (V2) requested but unavailable in runtime.\n");
          break;
        }
        TfLiteGpuDelegateOptionsV2 delegate_options =
            runtime_.GpuDelegateV2OptionsDefault();
        delegate_options.experimental_flags |=
            TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
        TfLiteDelegate* created =
            runtime_.GpuDelegateV2Create(&delegate_options);
        if (!AttachDelegate(created, runtime_.GpuDelegateV2Delete, "GPU V2",
                            MP_DELEGATE_GPU_V2)) {
          if (!allow_delegate_fallback) {
            SetError("Failed to create GPU delegate for face detector because "
                     "delegate fallback is disabled.");
            return false;
          }
          MP_DETECT_LOGE(
              "Failed to create GPU delegate. Falling back to CPU.\n");
        }
        break;
      }
      case MP_DELEGATE_CPU:
      default:
        break;
    }

    interpreter_.reset(runtime_.InterpreterCreate(model_.get(), options_.get()));
    bool tensors_ready =
        interpreter_ &&
        runtime_.InterpreterAllocateTensors(interpreter_.get()) == kTfLiteOk;
    // A delegate can also fail after it is attached, while the interpreter
    // builds or allocates tensors (e.g. the GPU delegate rejecting a graph
    // with custom ops). Honor delegate fallback for that stage too.
    if (!tensors_ready && active_delegate_ != MP_DELEGATE_CPU &&
        allow_delegate_fallback) {
      MP_DETECT_LOGE(
          "Interpreter creation with the requested delegate failed. "
          "Falling back to CPU.\n");
      interpreter_.reset();
      options_.reset(runtime_.InterpreterOptionsCreate());
      delegate_.reset();
      if (!options_) {
        SetError("Failed to allocate interpreter options.");
        return false;
      }
      runtime_.InterpreterOptionsSetThreads(options_.get(), threads_);
      active_delegate_ = MP_DELEGATE_CPU;
      interpreter_.reset(
          runtime_.InterpreterCreate(model_.get(), options_.get()));
      tensors_ready =
          interpreter_ &&
          runtime_.InterpreterAllocateTensors(interpreter_.get()) == kTfLiteOk;
    }
    if (!interpreter_) {
      SetError("Failed to create interpreter.");
      return false;
    }
    if (!tensors_ready) {
      SetError("Tensor allocation failed.");
      return false;
    }

    if (runtime_.InterpreterGetInputTensorCount(interpreter_.get()) < 1) {
      SetError("Interpreter input tensor missing.");
      return false;
    }
    input_tensor_ = runtime_.InterpreterGetInputTensor(interpreter_.get(), 0);
    if (!input_tensor_) {
      SetError("Input tensor unavailable.");
      return false;
    }
    if (runtime_.TensorType(input_tensor_) != kTfLiteFloat32) {
      SetError("Face detector input must be float32.");
      return false;
    }
    if (runtime_.TensorNumDims(input_tensor_) != 4) {
      SetError("Expected NHWC input tensor layout.");
      return false;
    }
    const int batch = runtime_.TensorDim(input_tensor_, 0);
    input_height_ = runtime_.TensorDim(input_tensor_, 1);
    input_width_ = runtime_.TensorDim(input_tensor_, 2);
    const int channels = runtime_.TensorDim(input_tensor_, 3);
    if (batch != 1 || channels != 3) {
      SetError("Model expects 1xHxWx3 input.");
      return false;
    }
    input_buffer_.resize(
        static_cast<size_t>(input_height_ * input_width_ * channels));

    const int output_count =
        runtime_.InterpreterGetOutputTensorCount(interpreter_.get());
    if (output_count < 2) {
      SetError("Face detector expects at least 2 output tensors.");
      return false;
    }

    output_boxes_tensor_ = runtime_.InterpreterGetOutputTensor(interpreter_.get(), 0);
    output_scores_tensor_ =
        runtime_.InterpreterGetOutputTensor(interpreter_.get(), 1);
    if (!output_boxes_tensor_ || !output_scores_tensor_) {
      SetError("Detector outputs are unavailable.");
      return false;
    }
    if (runtime_.TensorType(output_boxes_tensor_) != kTfLiteFloat32 ||
        runtime_.TensorType(output_scores_tensor_) != kTfLiteFloat32) {
      SetError("Detector outputs must be float32.");
      return false;
    }

    boxes_count_ = 1;
    const int boxes_dims = runtime_.TensorNumDims(output_boxes_tensor_);
    for (int i = 0; i < boxes_dims; ++i) {
      boxes_count_ *= runtime_.TensorDim(output_boxes_tensor_, i);
    }
    scores_count_ = 1;
    const int scores_dims = runtime_.TensorNumDims(output_scores_tensor_);
    for (int i = 0; i < scores_dims; ++i) {
      scores_count_ *= runtime_.TensorDim(output_scores_tensor_, i);
    }
    if (boxes_count_ <= 0 || scores_count_ <= 0) {
      SetError("Detector outputs are empty.");
      return false;
    }

    boxes_buffer_.resize(static_cast<size_t>(boxes_count_));
    scores_buffer_.resize(static_cast<size_t>(scores_count_));
    if (boxes_count_ % 16 != 0) {
      SetError("Unexpected detector box tensor size.");
      return false;
    }
    num_boxes_ = boxes_count_ / 16;
    if (scores_count_ % num_boxes_ != 0) {
      SetError("Detector score tensor does not match box tensor.");
      return false;
    }
    num_classes_ = scores_count_ / num_boxes_;
    if (!ConfigureDecoder()) {
      return false;
    }
    if (static_cast<int>(anchors_.size()) != num_boxes_) {
      SetError("Anchor count does not match detector output tensor.");
      return false;
    }

    MP_DETECT_LOGI("Face detector initialized: input=%dx%d boxes=%d classes=%d\n",
                   input_width_, input_height_, num_boxes_, num_classes_);
    return true;
  }

  MpFaceDetectorResult* Process(const MpImage& image,
                                const MpNormalizedRect* override_rect,
                                int rotation_degrees = 0,
                                bool mirror_horizontal = false,
                                const MpRoiTransformOptions* roi_transform = nullptr) {
    if (!interpreter_) {
      SetError("Interpreter is not initialized.");
      return nullptr;
    }
    if (!image.data || image.width <= 0 || image.height <= 0 ||
        image.bytes_per_row <= 0) {
      SetError("Invalid image buffer.");
      return nullptr;
    }
    if (image.format != MP_PIXEL_FORMAT_RGBA &&
        image.format != MP_PIXEL_FORMAT_BGRA) {
      SetError("Unsupported pixel format. Use RGBA/BGRA.");
      return nullptr;
    }

    const int rot = NormalizeRotationDegrees(rotation_degrees);
    if (rot < 0) {
      SetError("rotation_degrees must be one of 0, 90, 180, 270.");
      return nullptr;
    }

    const int logical_width = (rot == 90 || rot == 270) ? image.height : image.width;
    const int logical_height = (rot == 90 || rot == 270) ? image.width : image.height;
    const MpNormalizedRect rect =
        override_rect ? SanitizeRect(*override_rect) : DefaultRect();

    const bool needs_transform = rot != 0 || mirror_horizontal;
    if (needs_transform) {
      if (!PreprocessRotated(image, rect, rot, mirror_horizontal, logical_width,
                             logical_height)) {
        return nullptr;
      }
    } else {
      if (!Preprocess(image, rect)) {
        return nullptr;
      }
    }

    return InvokeAndDecode(logical_width, logical_height, rect, roi_transform);
  }

  MpFaceDetectorResult* ProcessNv21(const MpNv21Image& image,
                                    const MpNormalizedRect* override_rect,
                                    int rotation_degrees = 0,
                                    bool mirror_horizontal = false,
                                    const MpRoiTransformOptions* roi_transform = nullptr) {
    if (!interpreter_) {
      SetError("Interpreter is not initialized.");
      return nullptr;
    }
    if (!image.y || !image.vu || image.width <= 0 || image.height <= 0 ||
        image.y_bytes_per_row <= 0 || image.vu_bytes_per_row <= 0) {
      SetError("Invalid NV21 image buffer.");
      return nullptr;
    }

    const int rot = NormalizeRotationDegrees(rotation_degrees);
    if (rot < 0) {
      SetError("rotation_degrees must be one of 0, 90, 180, 270.");
      return nullptr;
    }

    const int logical_width = (rot == 90 || rot == 270) ? image.height : image.width;
    const int logical_height =
        (rot == 90 || rot == 270) ? image.width : image.height;
    const MpNormalizedRect rect =
        override_rect ? SanitizeRect(*override_rect) : DefaultRect();

    const bool needs_transform = rot != 0 || mirror_horizontal;
    if (needs_transform) {
      if (!PreprocessNv21Rotated(image, rect, rot, mirror_horizontal,
                                 logical_width, logical_height)) {
        return nullptr;
      }
    } else {
      if (!PreprocessNv21(image, rect)) {
        return nullptr;
      }
    }

    return InvokeAndDecode(logical_width, logical_height, rect, roi_transform);
  }

  const char* last_error() const { return last_error_.c_str(); }

  MpDelegateType active_delegate() const { return active_delegate_; }

 private:
  struct TfLiteModelDeleter {
    TfLiteRuntime* runtime;
    void operator()(TfLiteModel* model) const {
      if (runtime && model) {
        runtime->ModelDelete(model);
      }
    }
  };

  struct TfLiteOptionsDeleter {
    TfLiteRuntime* runtime;
    void operator()(TfLiteInterpreterOptions* options) const {
      if (runtime && options) {
        runtime->InterpreterOptionsDelete(options);
      }
    }
  };

  struct TfLiteInterpreterDeleter {
    TfLiteRuntime* runtime;
    void operator()(TfLiteInterpreter* interpreter) const {
      if (runtime && interpreter) {
        runtime->InterpreterDelete(interpreter);
      }
    }
  };

  struct TfLiteDelegateDeleter {
    using DeleteFn = void (*)(TfLiteDelegate*);
    DeleteFn deleter = nullptr;
    void operator()(TfLiteDelegate* delegate) const {
      if (deleter && delegate) {
        deleter(delegate);
      }
    }
  };

  void Shutdown() {
    interpreter_.reset();
    options_.reset();
    model_.reset();
    delegate_.reset();
    runtime_.Release();
  }

  MpNormalizedRect DefaultRect() const {
    MpNormalizedRect rect;
    rect.x_center = 0.5f;
    rect.y_center = 0.5f;
    rect.width = 1.0f;
    rect.height = 1.0f;
    rect.rotation = 0.0f;
    return rect;
  }

  MpNormalizedRect SanitizeRect(MpNormalizedRect rect) const {
    if (!(rect.width > 0.f) || !(rect.height > 0.f)) {
      return DefaultRect();
    }
    rect.x_center = Clamp(rect.x_center, 0.0f, 1.0f);
    rect.y_center = Clamp(rect.y_center, 0.0f, 1.0f);
    rect.width = Clamp(rect.width, 0.1f, 2.0f);
    rect.height = Clamp(rect.height, 0.1f, 2.0f);
    rect.rotation = NormalizeAngle(rect.rotation);
    return rect;
  }

  static int NormalizeRotationDegrees(int rotation_degrees) {
    switch (rotation_degrees) {
      case 0:
      case 90:
      case 180:
      case 270:
        return rotation_degrees;
      default:
        return -1;
    }
  }

  MpFaceDetectorResult* InvokeAndDecode(int logical_width,
                                        int logical_height,
                                        const MpNormalizedRect& rect,
                                        const MpRoiTransformOptions* roi_transform = nullptr) {
    const size_t input_bytes = input_buffer_.size() * sizeof(float);
    if (runtime_.TensorCopyFromBuffer(input_tensor_, input_buffer_.data(),
                                      input_bytes) != kTfLiteOk) {
      SetError("Failed to copy detector input buffer.");
      return nullptr;
    }
    if (runtime_.InterpreterInvoke(interpreter_.get()) != kTfLiteOk) {
      SetError("Detector invocation failed.");
      return nullptr;
    }
    if (runtime_.TensorCopyToBuffer(output_boxes_tensor_, boxes_buffer_.data(),
                                    boxes_buffer_.size() * sizeof(float)) !=
        kTfLiteOk) {
      SetError("Unable to read detector box output.");
      return nullptr;
    }
    if (runtime_.TensorCopyToBuffer(output_scores_tensor_, scores_buffer_.data(),
                                    scores_buffer_.size() * sizeof(float)) !=
        kTfLiteOk) {
      SetError("Unable to read detector score output.");
      return nullptr;
    }

    std::vector<DecodedDetection> detections = DecodeDetections(rect);
    if (detections.empty()) {
      return BuildResult(logical_width, logical_height, detections, roi_transform);
    }

    std::sort(detections.begin(), detections.end(),
              [](const DecodedDetection& a, const DecodedDetection& b) {
                return a.score > b.score;
              });

    std::vector<DecodedDetection> selected;
    selected.reserve(std::min<int>(max_results_, detections.size()));
    std::vector<bool> merged(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
      if (merged[i]) {
        continue;
      }

      std::vector<const DecodedDetection*> group;
      group.push_back(&detections[i]);
      merged[i] = true;
      for (size_t j = i + 1; j < detections.size(); ++j) {
        if (merged[j]) {
          continue;
        }
        if (IntersectionOverUnion(detections[i], detections[j]) >
            min_suppression_threshold_) {
          group.push_back(&detections[j]);
          merged[j] = true;
        }
      }

      selected.push_back(WeightedMergeDetections(group));
      if (static_cast<int>(selected.size()) >= max_results_) {
        break;
      }
    }

    return BuildResult(logical_width, logical_height, selected, roi_transform);
  }

  std::vector<DecodedDetection> DecodeDetections(
      const MpNormalizedRect& /*rect*/) const {
    std::vector<DecodedDetection> detections;
    detections.reserve(num_boxes_);

    for (int box_index = 0; box_index < num_boxes_; ++box_index) {
      float max_score = scores_buffer_[box_index * num_classes_];
      for (int class_index = 1; class_index < num_classes_; ++class_index) {
        max_score = std::max(
            max_score, scores_buffer_[box_index * num_classes_ + class_index]);
      }
      const float score = Sigmoid(max_score);
      if (score < min_detection_confidence_) {
        continue;
      }

      const Anchor& anchor = anchors_[box_index];
      const float* raw = boxes_buffer_.data() + box_index * 16;

      // MediaPipe configures BlazeFace with reverse_output_order=true, which
      // means boxes/keypoints are decoded as x/y/w/h rather than y/x/h/w.
      const float x_center =
          raw[0] / x_scale_ * anchor.width + anchor.x_center;
      const float y_center =
          raw[1] / y_scale_ * anchor.height + anchor.y_center;
      const float width = raw[2] / w_scale_ * anchor.width;
      const float height = raw[3] / h_scale_ * anchor.height;
      if (!(width > 0.0f) || !(height > 0.0f)) {
        continue;
      }

      const float x_min = x_center - width * 0.5f;
      const float y_min = y_center - height * 0.5f;
      const float x_max = x_center + width * 0.5f;
      const float y_max = y_center + height * 0.5f;

      DecodedDetection detection;
      float min_x = static_cast<float>(last_image_width_);
      float min_y = static_cast<float>(last_image_height_);
      float max_x = 0.0f;
      float max_y = 0.0f;
      const float kBoxCorners[4][2] = {
          {x_min, y_min},
          {x_max, y_min},
          {x_min, y_max},
          {x_max, y_max},
      };
      for (const auto& corner : kBoxCorners) {
        float image_x = 0.0f;
        float image_y = 0.0f;
        if (!ProjectTensorPoint(corner[0], corner[1], &image_x, &image_y)) {
          continue;
        }
        min_x = std::min(min_x, image_x);
        min_y = std::min(min_y, image_y);
        max_x = std::max(max_x, image_x);
        max_y = std::max(max_y, image_y);
      }
      detection.left = Clamp(min_x, 0.0f, static_cast<float>(last_image_width_));
      detection.top = Clamp(min_y, 0.0f, static_cast<float>(last_image_height_));
      detection.right =
          Clamp(max_x, 0.0f, static_cast<float>(last_image_width_));
      detection.bottom =
          Clamp(max_y, 0.0f, static_cast<float>(last_image_height_));
      if (detection.right <= detection.left ||
          detection.bottom <= detection.top) {
        continue;
      }
      detection.left = Clamp(detection.left / last_image_width_, 0.0f, 1.0f);
      detection.top = Clamp(detection.top / last_image_height_, 0.0f, 1.0f);
      detection.right = Clamp(detection.right / last_image_width_, 0.0f, 1.0f);
      detection.bottom =
          Clamp(detection.bottom / last_image_height_, 0.0f, 1.0f);
      for (int keypoint_index = 0; keypoint_index < 6; ++keypoint_index) {
        const int coord_index = 4 + keypoint_index * 2;
        const float keypoint_x =
            raw[coord_index] / x_scale_ * anchor.width + anchor.x_center;
        const float keypoint_y =
            raw[coord_index + 1] / y_scale_ * anchor.height +
            anchor.y_center;
        float image_x = 0.0f;
        float image_y = 0.0f;
        if (ProjectTensorPoint(keypoint_x, keypoint_y, &image_x, &image_y)) {
          detection.keypoints[keypoint_index * 2] =
              Clamp(image_x / last_image_width_, 0.0f, 1.0f);
          detection.keypoints[keypoint_index * 2 + 1] =
              Clamp(image_y / last_image_height_, 0.0f, 1.0f);
        }
      }
      detection.score = score;
      detections.push_back(detection);
    }

    return detections;
  }

  MpFaceDetectorResult* BuildResult(
      int image_width,
      int image_height,
      const std::vector<DecodedDetection>& detections,
      const MpRoiTransformOptions* roi_transform = nullptr) {
    MpFaceDetectorResult* result = new MpFaceDetectorResult();
    if (!result) {
      SetError("Unable to allocate detector result.");
      return nullptr;
    }
    result->image_width = image_width;
    result->image_height = image_height;
    result->detections_count = static_cast<int32_t>(detections.size());
    result->detections = nullptr;
    if (detections.empty()) {
      return result;
    }

    result->detections = new MpDetection[detections.size()];
    if (!result->detections) {
      delete result;
      SetError("Unable to allocate detector detections buffer.");
      return nullptr;
    }
    for (size_t i = 0; i < detections.size(); ++i) {
      result->detections[i].left = detections[i].left;
      result->detections[i].top = detections[i].top;
      result->detections[i].right = detections[i].right;
      result->detections[i].bottom = detections[i].bottom;
      result->detections[i].score = detections[i].score;
      std::memcpy(result->detections[i].keypoints, detections[i].keypoints.data(),
                  sizeof(result->detections[i].keypoints));
      const RectInPixels face_rect = DetectionToRect(detections[i]);
      const float roi_sx = roi_transform ? roi_transform->scale_x : 1.0f;
      const float roi_sy = roi_transform ? roi_transform->scale_y : 1.0f;
      const float roi_dx = roi_transform ? roi_transform->shift_x : 0.0f;
      const float roi_dy = roi_transform ? roi_transform->shift_y : 0.0f;
      const RectInPixels expanded_face_rect =
          TransformRect(face_rect, roi_sx, roi_sy, true, roi_dx, roi_dy);
      result->detections[i].face_rect =
          ToNormalizedRect(face_rect, image_width, image_height);
      result->detections[i].expanded_face_rect =
          ToNormalizedRect(expanded_face_rect, image_width, image_height);
    }
    return result;
  }

  bool ConfigureDecoder() {
    if (input_width_ == 128 && input_height_ == 128 && num_boxes_ == 896) {
      anchors_ = GenerateShortRangeAnchors();
      x_scale_ = 128.0f;
      y_scale_ = 128.0f;
      w_scale_ = 128.0f;
      h_scale_ = 128.0f;
    } else if (input_width_ == 192 && input_height_ == 192 &&
               num_boxes_ == 2304) {
      anchors_ = GenerateFullRangeAnchors();
      x_scale_ = 192.0f;
      y_scale_ = 192.0f;
      w_scale_ = 192.0f;
      h_scale_ = 192.0f;
    } else {
      SetError("Unsupported face detector model shape.");
      return false;
    }
    if (anchors_.empty()) {
      SetError("Failed to generate detector anchors.");
      return false;
    }
    return true;
  }

  RectInPixels ToPixelRect(const MpNormalizedRect& rect,
                           int width,
                           int height) const {
    RectInPixels roi;
    roi.center_x = rect.x_center * static_cast<float>(width);
    roi.center_y = rect.y_center * static_cast<float>(height);
    roi.width = rect.width * static_cast<float>(width);
    roi.height = rect.height * static_cast<float>(height);
    roi.rotation = rect.rotation;
    if (roi.width <= 0.0f) {
      roi.width = static_cast<float>(width);
    }
    if (roi.height <= 0.0f) {
      roi.height = static_cast<float>(height);
    }
    return roi;
  }

  RectInPixels DetectionToRect(const DecodedDetection& detection) const {
    RectInPixels rect;
    rect.center_x =
        (detection.left + detection.right) * 0.5f * last_image_width_;
    rect.center_y =
        (detection.top + detection.bottom) * 0.5f * last_image_height_;
    rect.width = (detection.right - detection.left) * last_image_width_;
    rect.height = (detection.bottom - detection.top) * last_image_height_;

    const float left_eye_x = detection.keypoints[0] * last_image_width_;
    const float left_eye_y = detection.keypoints[1] * last_image_height_;
    const float right_eye_x = detection.keypoints[2] * last_image_width_;
    const float right_eye_y = detection.keypoints[3] * last_image_height_;
    const float dx = right_eye_x - left_eye_x;
    const float dy = right_eye_y - left_eye_y;
    if (std::abs(dx) > 1e-6f || std::abs(dy) > 1e-6f) {
      rect.rotation = NormalizeAngle(std::atan2(dy, dx));
    }
    return rect;
  }

  RectInPixels TransformRect(const RectInPixels& rect,
                             float scale_x,
                             float scale_y,
                             bool square_long = false,
                             float shift_x = 0.0f,
                             float shift_y = 0.0f) const {
    RectInPixels transformed = rect;
    if (square_long) {
      const float long_side =
          std::max(rect.width, rect.height);
      transformed.width = long_side;
      transformed.height = long_side;
    }

    const float cos_r = std::cos(rect.rotation);
    const float sin_r = std::sin(rect.rotation);
    transformed.center_x =
        rect.center_x + rect.width * shift_x * cos_r -
        rect.height * shift_y * sin_r;
    transformed.center_y =
        rect.center_y + rect.width * shift_x * sin_r +
        rect.height * shift_y * cos_r;
    transformed.width *= scale_x;
    transformed.height *= scale_y;
    return transformed;
  }

  TensorTransform CreateTensorTransform(const RectInPixels& roi) const {
    TensorTransform transform;
    transform.center_x = roi.center_x;
    transform.center_y = roi.center_y;
    transform.roi_width = roi.width;
    transform.roi_height = roi.height;
    transform.rotation = roi.rotation;
    transform.scale =
        std::min(static_cast<float>(input_width_) / roi.width,
                 static_cast<float>(input_height_) / roi.height);
    transform.scaled_width = roi.width * transform.scale;
    transform.scaled_height = roi.height * transform.scale;
    transform.pad_x =
        (static_cast<float>(input_width_) - transform.scaled_width) * 0.5f;
    transform.pad_y =
        (static_cast<float>(input_height_) - transform.scaled_height) * 0.5f;
    return transform;
  }

  void UpdateProjectionMatrix() {
    if (last_transform_.scale <= 0.0f || last_image_width_ <= 0 ||
        last_image_height_ <= 0) {
      last_projection_ = ProjectionMatrix{};
      return;
    }

    const float inv_image_width = 1.0f / last_image_width_;
    const float inv_image_height = 1.0f / last_image_height_;
    const float inv_scale = 1.0f / last_transform_.scale;
    const float cos_r = std::cos(last_transform_.rotation);
    const float sin_r = std::sin(last_transform_.rotation);
    const float half_roi_width = last_transform_.roi_width * 0.5f;
    const float half_roi_height = last_transform_.roi_height * 0.5f;

    const float bx =
        (-last_transform_.pad_x) * inv_scale - half_roi_width;
    const float by =
        (-last_transform_.pad_y) * inv_scale - half_roi_height;
    const float cx =
        cos_r * bx - sin_r * by + last_transform_.center_x;
    const float cy =
        sin_r * bx + cos_r * by + last_transform_.center_y;

    last_projection_.m00 =
        cos_r * input_width_ * inv_scale * inv_image_width;
    last_projection_.m01 =
        -sin_r * input_height_ * inv_scale * inv_image_width;
    last_projection_.m02 = cx * inv_image_width;
    last_projection_.m10 =
        sin_r * input_width_ * inv_scale * inv_image_height;
    last_projection_.m11 =
        cos_r * input_height_ * inv_scale * inv_image_height;
    last_projection_.m12 = cy * inv_image_height;
  }

  bool ProjectTensorPoint(float x_norm,
                          float y_norm,
                          float* image_x,
                          float* image_y) const {
    if (!image_x || !image_y) {
      return false;
    }
    *image_x = (last_projection_.m00 * x_norm + last_projection_.m01 * y_norm +
                last_projection_.m02) *
               last_image_width_;
    *image_y = (last_projection_.m10 * x_norm + last_projection_.m11 * y_norm +
                last_projection_.m12) *
               last_image_height_;
    return true;
  }

  bool Preprocess(const MpImage& image, const MpNormalizedRect& rect) {
    const RectInPixels roi = ToPixelRect(rect, image.width, image.height);
    if (roi.width <= 0.f || roi.height <= 0.f) {
      SetError("Invalid ROI dimension.");
      return false;
    }
    last_transform_ = CreateTensorTransform(roi);
    last_image_width_ = image.width;
    last_image_height_ = image.height;
    UpdateProjectionMatrix();
    const float cos_r = std::cos(last_transform_.rotation);
    const float sin_r = std::sin(last_transform_.rotation);

    float* dst = input_buffer_.data();
    size_t offset = 0;
    for (int y = 0; y < input_height_; ++y) {
      const float tensor_y = static_cast<float>(y) + 0.5f;
      for (int x = 0; x < input_width_; ++x) {
        const float tensor_x = static_cast<float>(x) + 0.5f;
        const float roi_x =
            (tensor_x - last_transform_.pad_x) / last_transform_.scale;
        const float roi_y =
            (tensor_y - last_transform_.pad_y) / last_transform_.scale;
        if (roi_x < 0.0f || roi_y < 0.0f || roi_x > last_transform_.roi_width ||
            roi_y > last_transform_.roi_height) {
          dst[offset++] = -1.0f;
          dst[offset++] = -1.0f;
          dst[offset++] = -1.0f;
          continue;
        }
        const float local_x = roi_x - last_transform_.roi_width * 0.5f;
        const float local_y = roi_y - last_transform_.roi_height * 0.5f;
        const float source_x =
            cos_r * local_x - sin_r * local_y + last_transform_.center_x;
        const float source_y =
            sin_r * local_x + cos_r * local_y + last_transform_.center_y;
        const RgbPixel pixel = BilinearSample(image, source_x, source_y);
        dst[offset++] = pixel.r / 127.5f - 1.0f;
        dst[offset++] = pixel.g / 127.5f - 1.0f;
        dst[offset++] = pixel.b / 127.5f - 1.0f;
      }
    }
    return true;
  }

  bool PreprocessRotated(const MpImage& image,
                         const MpNormalizedRect& rect,
                         int rotation_degrees,
                         bool mirror_horizontal,
                         int rotated_width,
                         int rotated_height) {
    const RectInPixels roi = ToPixelRect(rect, rotated_width, rotated_height);
    if (roi.width <= 0.f || roi.height <= 0.f) {
      SetError("Invalid ROI dimension.");
      return false;
    }
    last_transform_ = CreateTensorTransform(roi);
    last_image_width_ = rotated_width;
    last_image_height_ = rotated_height;
    UpdateProjectionMatrix();
    const float cos_r = std::cos(last_transform_.rotation);
    const float sin_r = std::sin(last_transform_.rotation);

    float* dst = input_buffer_.data();
    size_t offset = 0;
    for (int y = 0; y < input_height_; ++y) {
      const float tensor_y = static_cast<float>(y) + 0.5f;
      for (int x = 0; x < input_width_; ++x) {
        const float tensor_x = static_cast<float>(x) + 0.5f;
        const float roi_x =
            (tensor_x - last_transform_.pad_x) / last_transform_.scale;
        const float roi_y =
            (tensor_y - last_transform_.pad_y) / last_transform_.scale;
        if (roi_x < 0.0f || roi_y < 0.0f || roi_x > last_transform_.roi_width ||
            roi_y > last_transform_.roi_height) {
          dst[offset++] = -1.0f;
          dst[offset++] = -1.0f;
          dst[offset++] = -1.0f;
          continue;
        }
        const float local_x = roi_x - last_transform_.roi_width * 0.5f;
        const float local_y = roi_y - last_transform_.roi_height * 0.5f;
        const float source_x =
            cos_r * local_x - sin_r * local_y + last_transform_.center_x;
        const float source_y =
            sin_r * local_x + cos_r * local_y + last_transform_.center_y;
        const RgbPixel pixel = BilinearSampleRotated(
            image, source_x, source_y, rotation_degrees, mirror_horizontal,
            rotated_width, rotated_height);
        dst[offset++] = pixel.r / 127.5f - 1.0f;
        dst[offset++] = pixel.g / 127.5f - 1.0f;
        dst[offset++] = pixel.b / 127.5f - 1.0f;
      }
    }
    return true;
  }

  bool PreprocessNv21(const MpNv21Image& image, const MpNormalizedRect& rect) {
    const RectInPixels roi = ToPixelRect(rect, image.width, image.height);
    if (roi.width <= 0.f || roi.height <= 0.f) {
      SetError("Invalid ROI dimension.");
      return false;
    }
    last_transform_ = CreateTensorTransform(roi);
    last_image_width_ = image.width;
    last_image_height_ = image.height;
    UpdateProjectionMatrix();
    const float cos_r = std::cos(last_transform_.rotation);
    const float sin_r = std::sin(last_transform_.rotation);

    float* dst = input_buffer_.data();
    size_t offset = 0;
    for (int y = 0; y < input_height_; ++y) {
      const float tensor_y = static_cast<float>(y) + 0.5f;
      for (int x = 0; x < input_width_; ++x) {
        const float tensor_x = static_cast<float>(x) + 0.5f;
        const float roi_x =
            (tensor_x - last_transform_.pad_x) / last_transform_.scale;
        const float roi_y =
            (tensor_y - last_transform_.pad_y) / last_transform_.scale;
        if (roi_x < 0.0f || roi_y < 0.0f || roi_x > last_transform_.roi_width ||
            roi_y > last_transform_.roi_height) {
          dst[offset++] = -1.0f;
          dst[offset++] = -1.0f;
          dst[offset++] = -1.0f;
          continue;
        }
        const float local_x = roi_x - last_transform_.roi_width * 0.5f;
        const float local_y = roi_y - last_transform_.roi_height * 0.5f;
        const float source_x =
            cos_r * local_x - sin_r * local_y + last_transform_.center_x;
        const float source_y =
            sin_r * local_x + cos_r * local_y + last_transform_.center_y;
        const RgbPixel pixel = BilinearSampleNv21(image, source_x, source_y);
        dst[offset++] = pixel.r / 127.5f - 1.0f;
        dst[offset++] = pixel.g / 127.5f - 1.0f;
        dst[offset++] = pixel.b / 127.5f - 1.0f;
      }
    }
    return true;
  }

  bool PreprocessNv21Rotated(const MpNv21Image& image,
                             const MpNormalizedRect& rect,
                             int rotation_degrees,
                             bool mirror_horizontal,
                             int rotated_width,
                             int rotated_height) {
    const RectInPixels roi = ToPixelRect(rect, rotated_width, rotated_height);
    if (roi.width <= 0.f || roi.height <= 0.f) {
      SetError("Invalid ROI dimension.");
      return false;
    }
    last_transform_ = CreateTensorTransform(roi);
    last_image_width_ = rotated_width;
    last_image_height_ = rotated_height;
    UpdateProjectionMatrix();
    const float cos_r = std::cos(last_transform_.rotation);
    const float sin_r = std::sin(last_transform_.rotation);

    float* dst = input_buffer_.data();
    size_t offset = 0;
    for (int y = 0; y < input_height_; ++y) {
      const float tensor_y = static_cast<float>(y) + 0.5f;
      for (int x = 0; x < input_width_; ++x) {
        const float tensor_x = static_cast<float>(x) + 0.5f;
        const float roi_x =
            (tensor_x - last_transform_.pad_x) / last_transform_.scale;
        const float roi_y =
            (tensor_y - last_transform_.pad_y) / last_transform_.scale;
        if (roi_x < 0.0f || roi_y < 0.0f || roi_x > last_transform_.roi_width ||
            roi_y > last_transform_.roi_height) {
          dst[offset++] = -1.0f;
          dst[offset++] = -1.0f;
          dst[offset++] = -1.0f;
          continue;
        }
        const float local_x = roi_x - last_transform_.roi_width * 0.5f;
        const float local_y = roi_y - last_transform_.roi_height * 0.5f;
        const float source_x =
            cos_r * local_x - sin_r * local_y + last_transform_.center_x;
        const float source_y =
            sin_r * local_x + cos_r * local_y + last_transform_.center_y;
        const RgbPixel pixel = BilinearSampleNv21Rotated(
            image, source_x, source_y, rotation_degrees, mirror_horizontal,
            rotated_width, rotated_height);
        dst[offset++] = pixel.r / 127.5f - 1.0f;
        dst[offset++] = pixel.g / 127.5f - 1.0f;
        dst[offset++] = pixel.b / 127.5f - 1.0f;
      }
    }
    return true;
  }

  RgbPixel BilinearSample(const MpImage& image, float x, float y) const {
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(image.width - 1) ||
        y > static_cast<float>(image.height - 1)) {
      return {};
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const RgbPixel p00 = ReadPixel(image, x0, y0);
    const RgbPixel p10 = ReadPixel(image, x1, y0);
    const RgbPixel p01 = ReadPixel(image, x0, y1);
    const RgbPixel p11 = ReadPixel(image, x1, y1);
    return Lerp(Lerp(p00, p10, dx), Lerp(p01, p11, dx), dy);
  }

  RgbPixel BilinearSampleRotated(const MpImage& image,
                                 float x,
                                 float y,
                                 int rotation_degrees,
                                 bool mirror_horizontal,
                                 int rotated_width,
                                 int rotated_height) const {
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(rotated_width - 1) ||
        y > static_cast<float>(rotated_height - 1)) {
      return {};
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, rotated_width - 1);
    const int y1 = std::min(y0 + 1, rotated_height - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const RgbPixel p00 = ReadPixelRotated(
        image, x0, y0, rotation_degrees, mirror_horizontal, rotated_width);
    const RgbPixel p10 = ReadPixelRotated(
        image, x1, y0, rotation_degrees, mirror_horizontal, rotated_width);
    const RgbPixel p01 = ReadPixelRotated(
        image, x0, y1, rotation_degrees, mirror_horizontal, rotated_width);
    const RgbPixel p11 = ReadPixelRotated(
        image, x1, y1, rotation_degrees, mirror_horizontal, rotated_width);
    return Lerp(Lerp(p00, p10, dx), Lerp(p01, p11, dx), dy);
  }

  RgbPixel BilinearSampleNv21(const MpNv21Image& image,
                              float x,
                              float y) const {
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(image.width - 1) ||
        y > static_cast<float>(image.height - 1)) {
      return {};
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const RgbPixel p00 = ReadPixelNv21(image, x0, y0);
    const RgbPixel p10 = ReadPixelNv21(image, x1, y0);
    const RgbPixel p01 = ReadPixelNv21(image, x0, y1);
    const RgbPixel p11 = ReadPixelNv21(image, x1, y1);
    return Lerp(Lerp(p00, p10, dx), Lerp(p01, p11, dx), dy);
  }

  RgbPixel BilinearSampleNv21Rotated(const MpNv21Image& image,
                                     float x,
                                     float y,
                                     int rotation_degrees,
                                     bool mirror_horizontal,
                                     int rotated_width,
                                     int rotated_height) const {
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(rotated_width - 1) ||
        y > static_cast<float>(rotated_height - 1)) {
      return {};
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, rotated_width - 1);
    const int y1 = std::min(y0 + 1, rotated_height - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const RgbPixel p00 = ReadPixelNv21Rotated(
        image, x0, y0, rotation_degrees, mirror_horizontal, rotated_width);
    const RgbPixel p10 = ReadPixelNv21Rotated(
        image, x1, y0, rotation_degrees, mirror_horizontal, rotated_width);
    const RgbPixel p01 = ReadPixelNv21Rotated(
        image, x0, y1, rotation_degrees, mirror_horizontal, rotated_width);
    const RgbPixel p11 = ReadPixelNv21Rotated(
        image, x1, y1, rotation_degrees, mirror_horizontal, rotated_width);
    return Lerp(Lerp(p00, p10, dx), Lerp(p01, p11, dx), dy);
  }

  RgbPixel ReadPixel(const MpImage& image, int x, int y) const {
    const uint8_t* row =
        image.data + static_cast<size_t>(y) * image.bytes_per_row;
    const uint8_t* ptr = row + static_cast<size_t>(x) * 4;
    RgbPixel pixel;
    if (image.format == MP_PIXEL_FORMAT_RGBA) {
      pixel.r = static_cast<float>(ptr[0]);
      pixel.g = static_cast<float>(ptr[1]);
      pixel.b = static_cast<float>(ptr[2]);
    } else {
      pixel.r = static_cast<float>(ptr[2]);
      pixel.g = static_cast<float>(ptr[1]);
      pixel.b = static_cast<float>(ptr[0]);
    }
    return pixel;
  }

  RgbPixel ReadPixelNv21(const MpNv21Image& image, int x, int y) const {
    const uint8_t* y_row =
        image.y + static_cast<size_t>(y) * image.y_bytes_per_row;
    const uint8_t Y = y_row[static_cast<size_t>(x)];
    const int uv_x = x >> 1;
    const int uv_y = y >> 1;
    const uint8_t* vu_row =
        image.vu + static_cast<size_t>(uv_y) * image.vu_bytes_per_row;
    const size_t vu_index = static_cast<size_t>(uv_x) * 2;
    const uint8_t V = vu_row[vu_index];
    const uint8_t U = vu_row[vu_index + 1];

    const int C = static_cast<int>(Y) - 16;
    const int D = static_cast<int>(U) - 128;
    const int E = static_cast<int>(V) - 128;
    const int c = C < 0 ? 0 : C;

    RgbPixel pixel;
    pixel.r = static_cast<float>(ClampInt((298 * c + 409 * E + 128) >> 8, 0, 255));
    pixel.g = static_cast<float>(
        ClampInt((298 * c - 100 * D - 208 * E + 128) >> 8, 0, 255));
    pixel.b = static_cast<float>(ClampInt((298 * c + 516 * D + 128) >> 8, 0, 255));
    return pixel;
  }

  RgbPixel ReadPixelRotated(const MpImage& image,
                            int x_rot,
                            int y_rot,
                            int rotation_degrees,
                            bool mirror_horizontal,
                            int rotated_width) const {
    int x_raw = 0;
    int y_raw = 0;
    MapRotatedToRaw(x_rot, y_rot, rotation_degrees, mirror_horizontal,
                    image.width, image.height, rotated_width, x_raw, y_raw);
    return ReadPixel(image, x_raw, y_raw);
  }

  static inline void MapRotatedToRaw(int x_rot,
                                     int y_rot,
                                     int rotation_degrees,
                                     bool mirror_horizontal,
                                     int raw_width,
                                     int raw_height,
                                     int rotated_width,
                                     int& out_x,
                                     int& out_y) {
    int xr = x_rot;
    int yr = y_rot;
    if (mirror_horizontal) {
      xr = (rotated_width - 1) - xr;
    }
    switch (rotation_degrees) {
      case 90:
        out_x = yr;
        out_y = (raw_height - 1) - xr;
        break;
      case 180:
        out_x = (raw_width - 1) - xr;
        out_y = (raw_height - 1) - yr;
        break;
      case 270:
        out_x = (raw_width - 1) - yr;
        out_y = xr;
        break;
      case 0:
      default:
        out_x = xr;
        out_y = yr;
        break;
    }
    out_x = ClampInt(out_x, 0, raw_width - 1);
    out_y = ClampInt(out_y, 0, raw_height - 1);
  }

  RgbPixel ReadPixelNv21Rotated(const MpNv21Image& image,
                                int x_rot,
                                int y_rot,
                                int rotation_degrees,
                                bool mirror_horizontal,
                                int rotated_width) const {
    int x_raw = 0;
    int y_raw = 0;
    MapRotatedToRaw(x_rot, y_rot, rotation_degrees, mirror_horizontal,
                    image.width, image.height, rotated_width, x_raw, y_raw);
    return ReadPixelNv21(image, x_raw, y_raw);
  }

  RgbPixel Lerp(const RgbPixel& a, const RgbPixel& b, float t) const {
    const float blend = Clamp(t, 0.0f, 1.0f);
    RgbPixel out;
    out.r = a.r + (b.r - a.r) * blend;
    out.g = a.g + (b.g - a.g) * blend;
    out.b = a.b + (b.b - a.b) * blend;
    return out;
  }

  void SetError(const std::string& message) {
    last_error_ = message;
    MP_DETECT_LOGE("%s\n", message.c_str());
  }

  TfLiteRuntime runtime_;
  std::unique_ptr<TfLiteModel, TfLiteModelDeleter> model_{nullptr, {&runtime_}};
  std::unique_ptr<TfLiteInterpreterOptions, TfLiteOptionsDeleter> options_{
      nullptr, {&runtime_}};
  std::unique_ptr<TfLiteInterpreter, TfLiteInterpreterDeleter> interpreter_{
      nullptr, {&runtime_}};
  std::unique_ptr<TfLiteDelegate, TfLiteDelegateDeleter> delegate_{nullptr, {}};

  TfLiteTensor* input_tensor_ = nullptr;
  const TfLiteTensor* output_boxes_tensor_ = nullptr;
  const TfLiteTensor* output_scores_tensor_ = nullptr;

  int input_width_ = 0;
  int input_height_ = 0;
  int threads_ = 2;
  int boxes_count_ = 0;
  int scores_count_ = 0;
  int num_boxes_ = 0;
  int num_classes_ = 0;
  int max_results_ = 1;
  int last_image_width_ = 0;
  int last_image_height_ = 0;
  float min_detection_confidence_ = 0.5f;
  float min_suppression_threshold_ = 0.3f;
  float x_scale_ = 128.0f;
  float y_scale_ = 128.0f;
  float w_scale_ = 128.0f;
  float h_scale_ = 128.0f;
  MpDelegateType active_delegate_ = MP_DELEGATE_CPU;

  std::string last_error_;
  ProjectionMatrix last_projection_;
  TensorTransform last_transform_;
  std::vector<float> input_buffer_;
  std::vector<float> boxes_buffer_;
  std::vector<float> scores_buffer_;
  std::vector<Anchor> anchors_;
};

std::string& GlobalFaceDetectorError() {
  static thread_local std::string error;
  return error;
}

}  // namespace

struct MpFaceDetectorContext {
  FaceDetectorContext impl;
};

extern "C" {

MpFaceDetectorContext* mp_face_detector_create(
    const char* model_path,
    const MpFaceDetectorCreateOptions* options) {
  if (!model_path || std::strlen(model_path) == 0) {
    GlobalFaceDetectorError() = "Model path is empty.";
    return nullptr;
  }
  auto* context = new MpFaceDetectorContext();
  if (!context) {
    GlobalFaceDetectorError() = "Unable to allocate detector context.";
    return nullptr;
  }
  if (!context->impl.Initialize(model_path, options)) {
    GlobalFaceDetectorError() = context->impl.last_error();
    delete context;
    return nullptr;
  }
  GlobalFaceDetectorError().clear();
  return context;
}

void mp_face_detector_destroy(MpFaceDetectorContext* context) {
  delete context;
}

MpFaceDetectorResult* mp_face_detector_process(
    MpFaceDetectorContext* context,
    const MpImage* image,
    const MpNormalizedRect* override_rect,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal,
    const MpRoiTransformOptions* roi_transform) {
  if (!context || !image) {
    GlobalFaceDetectorError() = "Detector context or image is null.";
    return nullptr;
  }
  MpFaceDetectorResult* result =
      context->impl.Process(*image, override_rect, rotation_degrees,
                            mirror_horizontal != 0, roi_transform);
  if (!result) {
    GlobalFaceDetectorError() = context->impl.last_error();
  } else {
    GlobalFaceDetectorError().clear();
  }
  return result;
}

MpFaceDetectorResult* mp_face_detector_process_nv21(
    MpFaceDetectorContext* context,
    const MpNv21Image* image,
    const MpNormalizedRect* override_rect,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal,
    const MpRoiTransformOptions* roi_transform) {
  if (!context || !image) {
    GlobalFaceDetectorError() = "Detector context or image is null.";
    return nullptr;
  }
  MpFaceDetectorResult* result =
      context->impl.ProcessNv21(*image, override_rect, rotation_degrees,
                                mirror_horizontal != 0, roi_transform);
  if (!result) {
    GlobalFaceDetectorError() = context->impl.last_error();
  } else {
    GlobalFaceDetectorError().clear();
  }
  return result;
}

void mp_face_detector_release_result(MpFaceDetectorResult* result) {
  if (!result) {
    return;
  }
  delete[] result->detections;
  result->detections = nullptr;
  delete result;
}

const char* mp_face_detector_last_error(const MpFaceDetectorContext* context) {
  if (!context) {
    return "";
  }
  return context->impl.last_error();
}

const char* mp_face_detector_last_global_error(void) {
  return GlobalFaceDetectorError().c_str();
}

MpDelegateType mp_face_detector_active_delegate(
    const MpFaceDetectorContext* context) {
  if (!context) {
    return MP_DELEGATE_CPU;
  }
  return context->impl.active_delegate();
}

}  // extern "C"
