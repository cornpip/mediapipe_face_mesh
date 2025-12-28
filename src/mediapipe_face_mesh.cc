#include "mediapipe_face.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if defined(__OBJC__) && TARGET_OS_IPHONE
#import <Foundation/Foundation.h>
#endif
// Use the umbrella header from TensorFlowLiteC.framework on Apple platforms.
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
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tflite_runtime.h"

#if defined(__ANDROID__)
#include <android/log.h>
#define MP_LOG_TAG "MediapipeFaceMesh"
#define MP_LOGI(...) __android_log_print(ANDROID_LOG_INFO, MP_LOG_TAG, __VA_ARGS__)
#define MP_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, MP_LOG_TAG, __VA_ARGS__)
#else
#define MP_LOGI(...) std::fprintf(stdout, "[INFO] " __VA_ARGS__)
#define MP_LOGE(...) std::fprintf(stderr, "[ERROR] " __VA_ARGS__)
#endif

namespace {

struct RectInPixels {
  float center_x = 0.0f;
  float center_y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float rotation = 0.0f;
};

struct RgbPixel {
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
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

class FaceMeshContext {
 public:
  FaceMeshContext() = default;
  ~FaceMeshContext() { Shutdown(); }

  bool Initialize(const std::string& model_path,
                  const MpFaceMeshCreateOptions* options) {
    threads_ = 2;
    if (options && options->threads > 0) {
      threads_ = options->threads;
    }
    min_detection_confidence_ =
        (options && options->min_detection_confidence > 0.f)
            ? options->min_detection_confidence
            : 0.5f;
    min_tracking_confidence_ =
        (options && options->min_tracking_confidence > 0.f)
            ? options->min_tracking_confidence
            : 0.5f;
    smoothing_enabled_ = !options || options->enable_smoothing != 0;

    MP_LOGI("Initialize start: model=%s threads=%d\n", model_path.c_str(),
            threads_);

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
    auto AttachDelegate = [&](TfLiteDelegate* created,
                              TfLiteDelegateDeleter::DeleteFn deleter,
                              const char* name) {
      if (!created) {
        return false;
      }
      delegate_.get_deleter().deleter = deleter;
      delegate_.reset(created);
      runtime_.InterpreterOptionsAddDelegate(
          options_.get(),
          reinterpret_cast<TfLiteOpaqueDelegate*>(delegate_.get()));
      MP_LOGI("%s delegate enabled.\n", name);
      return true;
    };
    switch (delegate_choice) {
      case MP_DELEGATE_XNNPACK: {
        if (!runtime_.InterpreterOptionsAddDelegate ||
            !runtime_.XnnpackDelegateOptionsDefault ||
            !runtime_.XnnpackDelegateCreate || !runtime_.XnnpackDelegateDelete) {
          MP_LOGI("XNNPACK delegate requested but not available in runtime.\n");
          break;
        }
        TfLiteXNNPackDelegateOptions xnnpack_options =
            runtime_.XnnpackDelegateOptionsDefault();
        xnnpack_options.num_threads = threads_;
        TfLiteDelegate* created_delegate =
            runtime_.XnnpackDelegateCreate(&xnnpack_options);
        if (!AttachDelegate(created_delegate, runtime_.XnnpackDelegateDelete,
                            "XNNPACK")) {
          MP_LOGE("Failed to create XNNPACK delegate. Falling back to CPU.\n");
        }
        break;
      }
      case MP_DELEGATE_GPU_V2: {
        if (!runtime_.InterpreterOptionsAddDelegate ||
            !runtime_.GpuDelegateV2OptionsDefault ||
            !runtime_.GpuDelegateV2Create || !runtime_.GpuDelegateV2Delete) {
          MP_LOGI("GPU delegate (V2) requested but not available in runtime.\n");
          break;
        }
        TfLiteGpuDelegateOptionsV2 gpu_options =
            runtime_.GpuDelegateV2OptionsDefault();
        gpu_options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
        TfLiteDelegate* created_delegate =
            runtime_.GpuDelegateV2Create(&gpu_options);
        if (!AttachDelegate(created_delegate, runtime_.GpuDelegateV2Delete,
                            "GPU V2")) {
          MP_LOGE("Failed to create GPU delegate. Falling back to CPU.\n");
        }
        break;
      }
      case MP_DELEGATE_CPU:
      default:
        break;
    }

    interpreter_.reset(runtime_.InterpreterCreate(model_.get(), options_.get()));
    if (!interpreter_) {
      SetError("Failed to create interpreter.");
      return false;
    }

    if (runtime_.InterpreterAllocateTensors(interpreter_.get()) != kTfLiteOk) {
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
      SetError("Model input must be float32.");
      return false;
    }
    if (runtime_.TensorNumDims(input_tensor_) != 4) {
      SetError("Expected NHWC tensor layout.");
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
    input_buffer_.resize(static_cast<size_t>(input_height_ * input_width_ * channels));

    const int output_count =
        runtime_.InterpreterGetOutputTensorCount(interpreter_.get());
    if (output_count < 1) {
      SetError("Model outputs are missing.");
      return false;
    }
    output_landmarks_tensor_ =
        runtime_.InterpreterGetOutputTensor(interpreter_.get(), 0);
    if (!output_landmarks_tensor_) {
      SetError("Landmark tensor missing.");
      return false;
    }
    if (runtime_.TensorType(output_landmarks_tensor_) != kTfLiteFloat32) {
      SetError("Landmark tensor must be float32.");
      return false;
    }
    int total = 1;
    const int dims = runtime_.TensorNumDims(output_landmarks_tensor_);
    for (int i = 0; i < dims; ++i) {
      total *= runtime_.TensorDim(output_landmarks_tensor_, i);
    }
    if (total % 3 != 0) {
      SetError("Unexpected landmark size.");
      return false;
    }
    output_landmark_count_ = total / 3;
    landmarks_buffer_.resize(static_cast<size_t>(total));

    if (output_count > 1) {
      output_score_tensor_ =
          runtime_.InterpreterGetOutputTensor(interpreter_.get(), 1);
      if (output_score_tensor_ &&
          runtime_.TensorType(output_score_tensor_) != kTfLiteFloat32) {
        output_score_tensor_ = nullptr;
      }
    }

    roi_ = DefaultRect();
    has_valid_rect_ = true;
    MP_LOGI("Initialize success\n");
    return true;
  }

  /// RGBA/BGRA
  MpFaceMeshResult* Process(const MpImage& image,
                            const MpNormalizedRect* override_rect,
                            int rotation_degrees = 0,
                            bool mirror_horizontal = false) {
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

    if (rot != last_rotation_degrees_ ||
        mirror_horizontal != last_mirror_horizontal_) {
      has_valid_rect_ = false;
      last_rotation_degrees_ = rot;
      last_mirror_horizontal_ = mirror_horizontal;
    }

    const int logical_width = (rot == 90 || rot == 270) ? image.height : image.width;
    const int logical_height =
        (rot == 90 || rot == 270) ? image.width : image.height;

    MpNormalizedRect rect;
    if (override_rect) {
      rect = SanitizeRect(*override_rect);
      has_valid_rect_ = true;
    } else if (has_valid_rect_) {
      rect = roi_;
    } else {
      rect = DefaultRect();
    }

    const bool needs_transform = rot != 0 || mirror_horizontal;
    if (needs_transform) {
      if (!PreprocessRotated(image, rect, rot, mirror_horizontal,
                             logical_width, logical_height)) {
        return nullptr;
      }
    } else {
      if (!Preprocess(image, rect)) {
        return nullptr;
      }
    }

    const size_t bytes = input_buffer_.size() * sizeof(float);
    if (runtime_.TensorCopyFromBuffer(input_tensor_, input_buffer_.data(),
                                      bytes) != kTfLiteOk) {
      SetError("Failed to copy input buffer.");
      return nullptr;
    }

    if (runtime_.InterpreterInvoke(interpreter_.get()) != kTfLiteOk) {
      SetError("Interpreter invocation failed.");
      return nullptr;
    }

    if (runtime_.TensorCopyToBuffer(output_landmarks_tensor_,
                                    landmarks_buffer_.data(),
                                    landmarks_buffer_.size() * sizeof(float)) !=
        kTfLiteOk) {
      SetError("Unable to read landmark output.");
      return nullptr;
    }

    float score = 1.0f;
    if (output_score_tensor_) {
      if (runtime_.TensorCopyToBuffer(output_score_tensor_, &score,
                                      sizeof(float)) != kTfLiteOk) {
        SetError("Unable to read confidence output.");
        return nullptr;
      }
    }

    MpFaceMeshResult* result =
        BuildResultFromSize(logical_width, logical_height, rect, score);
    if (!result) {
      return nullptr;
    }

    // Debug: log raw landmark ranges before normalization.
    if (!landmarks_buffer_.empty()) {
      float min_x = landmarks_buffer_[0];
      float max_x = landmarks_buffer_[0];
      float min_y = landmarks_buffer_[1];
      float max_y = landmarks_buffer_[1];
      for (int i = 0; i < output_landmark_count_; ++i) {
        const float rx = landmarks_buffer_[i * 3];
        const float ry = landmarks_buffer_[i * 3 + 1];
        min_x = std::min(min_x, rx);
        max_x = std::max(max_x, rx);
        min_y = std::min(min_y, ry);
        max_y = std::max(max_y, ry);
      }
      MP_LOGI("Raw landmarks: count=%d min_x=%.3f max_x=%.3f min_y=%.3f max_y=%.3f\n",
              output_landmark_count_, min_x, max_x, min_y, max_y);
    }

    if (!override_rect) {
      UpdateTrackingState(*result, score);
    } else {
      roi_ = rect;
      has_valid_rect_ = true;
    }

    return result;
  }

  MpFaceMeshResult* ProcessNv21(const MpNv21Image& image,
                               const MpNormalizedRect* override_rect,
                               int rotation_degrees = 0,
                               bool mirror_horizontal = false) {
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

    // Reset tracking state when the logical coordinate system changes.
    if (rot != last_rotation_degrees_ ||
        mirror_horizontal != last_mirror_horizontal_) {
      has_valid_rect_ = false;
      last_rotation_degrees_ = rot;
      last_mirror_horizontal_ = mirror_horizontal;
    }

    const int logical_width = (rot == 90 || rot == 270) ? image.height : image.width;
    const int logical_height = (rot == 90 || rot == 270) ? image.width : image.height;

    MpNormalizedRect rect;
    if (override_rect) {
      rect = SanitizeRect(*override_rect);
      has_valid_rect_ = true;
    } else if (has_valid_rect_) {
      rect = roi_;
    } else {
      rect = DefaultRect();
    }

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

    const size_t bytes = input_buffer_.size() * sizeof(float);
    if (runtime_.TensorCopyFromBuffer(input_tensor_, input_buffer_.data(),
                                      bytes) != kTfLiteOk) {
      SetError("Failed to copy input buffer.");
      return nullptr;
    }

    if (runtime_.InterpreterInvoke(interpreter_.get()) != kTfLiteOk) {
      SetError("Interpreter invocation failed.");
      return nullptr;
    }

    if (runtime_.TensorCopyToBuffer(output_landmarks_tensor_,
                                    landmarks_buffer_.data(),
                                    landmarks_buffer_.size() * sizeof(float)) !=
        kTfLiteOk) {
      SetError("Unable to read landmark output.");
      return nullptr;
    }

    float score = 1.0f;
    if (output_score_tensor_) {
      if (runtime_.TensorCopyToBuffer(output_score_tensor_, &score,
                                      sizeof(float)) != kTfLiteOk) {
        SetError("Unable to read confidence output.");
        return nullptr;
      }
    }

    MpFaceMeshResult* result =
        BuildResultFromSize(logical_width, logical_height, rect, score);
    if (!result) {
      return nullptr;
    }

    if (!override_rect) {
      UpdateTrackingState(*result, score);
    } else {
      roi_ = rect;
      has_valid_rect_ = true;
    }

    return result;
  }

  const char* last_error() const { return last_error_.c_str(); }

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

  bool Preprocess(const MpImage& image, const MpNormalizedRect& rect) {
    const RectInPixels roi = ToPixelRect(rect, image.width, image.height);
    if (roi.width <= 0.f || roi.height <= 0.f) {
      SetError("Invalid ROI dimension.");
      return false;
    }
    const float cos_r = std::cos(roi.rotation);
    const float sin_r = std::sin(roi.rotation);
    const float half_w = roi.width * 0.5f;
    const float half_h = roi.height * 0.5f;

    float* dst = input_buffer_.data();
    const int target_w = input_width_;
    const int target_h = input_height_;

    size_t offset = 0;
    for (int y = 0; y < target_h; ++y) {
      const float ny =
          ((static_cast<float>(y) + 0.5f) / static_cast<float>(target_h) - 0.5f) *
          2.0f;
      for (int x = 0; x < target_w; ++x) {
        const float nx = ((static_cast<float>(x) + 0.5f) /
                              static_cast<float>(target_w) -
                          0.5f) *
                         2.0f;
        const float rx = nx * half_w;
        const float ry = ny * half_h;
        const float source_x = cos_r * rx - sin_r * ry + roi.center_x;
        const float source_y = sin_r * rx + cos_r * ry + roi.center_y;
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
    const float cos_r = std::cos(roi.rotation);
    const float sin_r = std::sin(roi.rotation);
    const float half_w = roi.width * 0.5f;
    const float half_h = roi.height * 0.5f;

    float* dst = input_buffer_.data();
    const int target_w = input_width_;
    const int target_h = input_height_;

    size_t offset = 0;
    for (int y = 0; y < target_h; ++y) {
      const float ny =
          ((static_cast<float>(y) + 0.5f) / static_cast<float>(target_h) -
           0.5f) *
          2.0f;
      for (int x = 0; x < target_w; ++x) {
        const float nx =
            ((static_cast<float>(x) + 0.5f) / static_cast<float>(target_w) -
             0.5f) *
            2.0f;
        const float rx = nx * half_w;
        const float ry = ny * half_h;
        const float source_x = cos_r * rx - sin_r * ry + roi.center_x;
        const float source_y = sin_r * rx + cos_r * ry + roi.center_y;
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
    const float cos_r = std::cos(roi.rotation);
    const float sin_r = std::sin(roi.rotation);
    const float half_w = roi.width * 0.5f;
    const float half_h = roi.height * 0.5f;

    float* dst = input_buffer_.data();
    const int target_w = input_width_;
    const int target_h = input_height_;

    size_t offset = 0;
    for (int y = 0; y < target_h; ++y) {
      const float ny =
          ((static_cast<float>(y) + 0.5f) / static_cast<float>(target_h) -
           0.5f) *
          2.0f;
      for (int x = 0; x < target_w; ++x) {
        const float nx =
            ((static_cast<float>(x) + 0.5f) / static_cast<float>(target_w) -
             0.5f) *
            2.0f;
        const float rx = nx * half_w;
        const float ry = ny * half_h;
        const float source_x = cos_r * rx - sin_r * ry + roi.center_x;
        const float source_y = sin_r * rx + cos_r * ry + roi.center_y;
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
    const float cos_r = std::cos(roi.rotation);
    const float sin_r = std::sin(roi.rotation);
    const float half_w = roi.width * 0.5f;
    const float half_h = roi.height * 0.5f;

    float* dst = input_buffer_.data();
    const int target_w = input_width_;
    const int target_h = input_height_;

    size_t offset = 0;
    for (int y = 0; y < target_h; ++y) {
      const float ny =
          ((static_cast<float>(y) + 0.5f) / static_cast<float>(target_h) -
           0.5f) *
          2.0f;
      for (int x = 0; x < target_w; ++x) {
        const float nx =
            ((static_cast<float>(x) + 0.5f) / static_cast<float>(target_w) -
             0.5f) *
            2.0f;
        const float rx = nx * half_w;
        const float ry = ny * half_h;
        const float source_x = cos_r * rx - sin_r * ry + roi.center_x;
        const float source_y = sin_r * rx + cos_r * ry + roi.center_y;
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

    const RgbPixel top = Lerp(p00, p10, dx);
    const RgbPixel bottom = Lerp(p01, p11, dx);
    return Lerp(top, bottom, dy);
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

    const RgbPixel top = Lerp(p00, p10, dx);
    const RgbPixel bottom = Lerp(p01, p11, dx);
    return Lerp(top, bottom, dy);
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

    const RgbPixel top = Lerp(p00, p10, dx);
    const RgbPixel bottom = Lerp(p01, p11, dx);
    return Lerp(top, bottom, dy);
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

    const RgbPixel top = Lerp(p00, p10, dx);
    const RgbPixel bottom = Lerp(p01, p11, dx);
    return Lerp(top, bottom, dy);
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

    const int r = (298 * c + 409 * E + 128) >> 8;
    const int g = (298 * c - 100 * D - 208 * E + 128) >> 8;
    const int b = (298 * c + 516 * D + 128) >> 8;

    RgbPixel pixel;
    pixel.r = static_cast<float>(ClampInt(r, 0, 255));
    pixel.g = static_cast<float>(ClampInt(g, 0, 255));
    pixel.b = static_cast<float>(ClampInt(b, 0, 255));
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
                    image.width, image.height, rotated_width,
                    x_raw, y_raw);
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

  MpFaceMeshResult* BuildResult(const MpImage& image,
                                const MpNormalizedRect& rect,
                                float score) {
    return BuildResultFromSize(image.width, image.height, rect, score);
  }

  MpFaceMeshResult* BuildResultFromSize(int width,
                                        int height,
                                        const MpNormalizedRect& rect,
                                        float score) {
    auto* result = new MpFaceMeshResult();
    if (!result) {
      SetError("Unable to allocate result.");
      return nullptr;
    }
    result->landmarks_count = output_landmark_count_;
    result->landmarks = new MpLandmark[output_landmark_count_];
    if (!result->landmarks) {
      SetError("Unable to allocate landmarks buffer.");
      delete result;
      return nullptr;
    }
    result->rect = rect;
    result->score = score;
    result->image_width = width;
    result->image_height = height;

    const RectInPixels roi = ToPixelRect(rect, width, height);
    const float cos_r = std::cos(roi.rotation);
    const float sin_r = std::sin(roi.rotation);
    const float half_w = roi.width * 0.5f;
    const float half_h = roi.height * 0.5f;
    const float input_w = std::max(1, input_width_);
    const float input_h = std::max(1, input_height_);

    for (int i = 0; i < output_landmark_count_; ++i) {
      float raw_x = landmarks_buffer_[i * 3];
      float raw_y = landmarks_buffer_[i * 3 + 1];
      float raw_z = landmarks_buffer_[i * 3 + 2];

      // Some models emit normalized [0,1], others emit pixel coordinates in
      // input resolution. If values are outside [0,1], normalize using input
      // tensor size.
      if (raw_x > 1.0f || raw_y > 1.0f || raw_x < 0.0f || raw_y < 0.0f) {
        raw_x = raw_x / input_w;
        raw_y = raw_y / input_h;
        raw_z = raw_z / input_w;
      }

      const float nx = (raw_x - 0.5f) * 2.0f;
      const float ny = (raw_y - 0.5f) * 2.0f;
      const float rx = nx * half_w;
      const float ry = ny * half_h;

      const float abs_x = cos_r * rx - sin_r * ry + roi.center_x;
      const float abs_y = sin_r * rx + cos_r * ry + roi.center_y;
      const float abs_z = raw_z * roi.width;

      MpLandmark& landmark = result->landmarks[i];
      landmark.x =
          Clamp(abs_x / static_cast<float>(width), -0.5f, 1.5f);
      landmark.y =
          Clamp(abs_y / static_cast<float>(height), -0.5f, 1.5f);
      landmark.z = abs_z / static_cast<float>(width);
    }
    return result;
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

  void UpdateTrackingState(const MpFaceMeshResult& result, float score) {
    const float threshold =
        has_valid_rect_ ? min_tracking_confidence_ : min_detection_confidence_;
    if (score < threshold) {
      return;
    }
    const MpNormalizedRect target =
        RectFromLandmarks(result.landmarks, result.landmarks_count);
    MpNormalizedRect updated = target;
    if (has_valid_rect_ && smoothing_enabled_) {
      updated = SmoothRect(roi_, target);
    }
    roi_ = SanitizeRect(updated);
    has_valid_rect_ = true;
  }

  MpNormalizedRect RectFromLandmarks(const MpLandmark* landmarks,
                                     int count) const {
    if (!landmarks || count <= 0) {
      return DefaultRect();
    }
    float min_x = 1.0f;
    float min_y = 1.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
    for (int i = 0; i < count; ++i) {
      min_x = std::min(min_x, landmarks[i].x);
      min_y = std::min(min_y, landmarks[i].y);
      max_x = std::max(max_x, landmarks[i].x);
      max_y = std::max(max_y, landmarks[i].y);
    }
    float width = max_x - min_x;
    float height = max_y - min_y;
    if (width < 1e-4f || height < 1e-4f) {
      return DefaultRect();
    }
    const float size = std::max(width, height) * 1.5f;
    MpNormalizedRect rect;
    rect.x_center = Clamp((min_x + max_x) * 0.5f, 0.0f, 1.0f);
    rect.y_center = Clamp((min_y + max_y) * 0.5f, 0.0f, 1.0f);
    rect.width = Clamp(size, 0.1f, 1.2f);
    rect.height = rect.width;
    rect.rotation = EstimateRotation(landmarks, count);
    return rect;
  }

  MpNormalizedRect SmoothRect(const MpNormalizedRect& current,
                              const MpNormalizedRect& target) const {
    constexpr float kAlpha = 0.8f;
    MpNormalizedRect rect;
    rect.x_center = current.x_center * kAlpha + target.x_center * (1.0f - kAlpha);
    rect.y_center = current.y_center * kAlpha + target.y_center * (1.0f - kAlpha);
    rect.width = current.width * kAlpha + target.width * (1.0f - kAlpha);
    rect.height = current.height * kAlpha + target.height * (1.0f - kAlpha);
    const float delta =
        NormalizeAngle(target.rotation - current.rotation) * (1.0f - kAlpha);
    rect.rotation = NormalizeAngle(current.rotation + delta);
    return rect;
  }

  float EstimateRotation(const MpLandmark* landmarks, int count) const {
    const int left_eye_index = 263;
    const int right_eye_index = 33;
    if (count <= left_eye_index || count <= right_eye_index) {
      return 0.0f;
    }
    const MpLandmark& left = landmarks[left_eye_index];
    const MpLandmark& right = landmarks[right_eye_index];
    const float dx = left.x - right.x;
    const float dy = left.y - right.y;
    if (std::abs(dx) < 1e-5f && std::abs(dy) < 1e-5f) {
      return 0.0f;
    }
    return std::atan2(dy, dx);
  }

  void SetError(const std::string& message) {
    last_error_ = message;
    MP_LOGE("%s\n", message.c_str());
  }

  TfLiteRuntime runtime_;
  std::unique_ptr<TfLiteModel, TfLiteModelDeleter> model_{nullptr, {&runtime_}};
  std::unique_ptr<TfLiteInterpreterOptions, TfLiteOptionsDeleter> options_{
      nullptr, {&runtime_}};
  std::unique_ptr<TfLiteInterpreter, TfLiteInterpreterDeleter> interpreter_{
      nullptr, {&runtime_}};
  std::unique_ptr<TfLiteDelegate, TfLiteDelegateDeleter> delegate_{nullptr, {}};

  TfLiteTensor* input_tensor_ = nullptr;
  const TfLiteTensor* output_landmarks_tensor_ = nullptr;
  const TfLiteTensor* output_score_tensor_ = nullptr;

  int input_width_ = 0;
  int input_height_ = 0;
  int output_landmark_count_ = 0;

  int threads_ = 2;
  float min_detection_confidence_ = 0.5f;
  float min_tracking_confidence_ = 0.5f;
  bool smoothing_enabled_ = true;

  std::vector<float> input_buffer_;
  std::vector<float> landmarks_buffer_;

  MpNormalizedRect roi_;
  bool has_valid_rect_ = false;
  int last_rotation_degrees_ = 0;
  bool last_mirror_horizontal_ = false;
  std::string last_error_;
};

thread_local std::string g_last_global_error;

void SetGlobalError(const std::string& message) {
  g_last_global_error = message;
}

}  // namespace

struct MpFaceMeshContext {
  FaceMeshContext impl;
};

extern "C" {

FFI_PLUGIN_EXPORT MpFaceMeshContext* mp_face_mesh_create(
    const char* model_path,
    const MpFaceMeshCreateOptions* options) {
  if (!model_path) {
    SetGlobalError("Model path is null.");
    return nullptr;
  }
  auto* context = new MpFaceMeshContext();
  if (!context) {
    SetGlobalError("Unable to allocate context.");
    return nullptr;
  }
  if (!context->impl.Initialize(model_path, options)) {
    SetGlobalError(context->impl.last_error());
    delete context;
    return nullptr;
  }
  return context;
}

FFI_PLUGIN_EXPORT void mp_face_mesh_destroy(MpFaceMeshContext* context) {
  delete context;
}

FFI_PLUGIN_EXPORT MpFaceMeshResult* mp_face_mesh_process(
    MpFaceMeshContext* context,
    const MpImage* image,
    const MpNormalizedRect* override_rect,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal) {
  if (!context) {
    SetGlobalError("Context is null.");
    return nullptr;
  }
  if (!image) {
    SetGlobalError("Image is null.");
    return nullptr;
  }
  return context->impl.Process(*image, override_rect, rotation_degrees,
                               mirror_horizontal != 0);
}

FFI_PLUGIN_EXPORT MpFaceMeshResult* mp_face_mesh_process_nv21(
    MpFaceMeshContext* context,
    const MpNv21Image* image,
    const MpNormalizedRect* override_rect,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal) {
  if (!context) {
    SetGlobalError("Context is null.");
    return nullptr;
  }
  if (!image) {
    SetGlobalError("Image is null.");
    return nullptr;
  }
  return context->impl.ProcessNv21(*image, override_rect, rotation_degrees,
                                   mirror_horizontal != 0);
}

FFI_PLUGIN_EXPORT void mp_face_mesh_release_result(MpFaceMeshResult* result) {
  if (!result) {
    return;
  }
  delete[] result->landmarks;
  result->landmarks = nullptr;
  delete result;
}

FFI_PLUGIN_EXPORT const char* mp_face_mesh_last_error(
    const MpFaceMeshContext* context) {
  if (!context) {
    return nullptr;
  }
  return context->impl.last_error();
}

FFI_PLUGIN_EXPORT const char* mp_face_mesh_last_global_error(void) {
  return g_last_global_error.c_str();
}

}  // extern "C"
