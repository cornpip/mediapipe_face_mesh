#include "mediapipe_face_mesh.h"

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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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

class TfLiteRuntime {
 public:
  using ModelCreateFromFileFn = TfLiteModel* (*)(const char*);
  using ModelDeleteFn = void (*)(TfLiteModel*);
  using InterpreterOptionsCreateFn = TfLiteInterpreterOptions* (*)();
  using InterpreterOptionsDeleteFn = void (*)(TfLiteInterpreterOptions*);
  using InterpreterOptionsSetThreadsFn = void (*)(TfLiteInterpreterOptions*, int32_t);
  using InterpreterCreateFn =
      TfLiteInterpreter* (*)(const TfLiteModel*, const TfLiteInterpreterOptions*);
  using InterpreterDeleteFn = void (*)(TfLiteInterpreter*);
  using InterpreterAllocateTensorsFn = TfLiteStatus (*)(TfLiteInterpreter*);
  using InterpreterInvokeFn = TfLiteStatus (*)(TfLiteInterpreter*);
  using InterpreterGetInputTensorFn = TfLiteTensor* (*)(TfLiteInterpreter*, int32_t);
  using InterpreterGetOutputTensorFn =
      const TfLiteTensor* (*)(const TfLiteInterpreter*, int32_t);
  using InterpreterGetInputTensorCountFn = int32_t (*)(const TfLiteInterpreter*);
  using InterpreterGetOutputTensorCountFn = int32_t (*)(const TfLiteInterpreter*);
  using TensorTypeFn = TfLiteType (*)(const TfLiteTensor*);
  using TensorNumDimsFn = int (*)(const TfLiteTensor*);
  using TensorDimFn = int (*)(const TfLiteTensor*, int);
  using TensorByteSizeFn = size_t (*)(const TfLiteTensor*);
  using TensorDataFn = void* (*)(const TfLiteTensor*);
  using TensorCopyFromBufferFn =
      TfLiteStatus (*)(TfLiteTensor*, const void*, size_t);
  using TensorCopyToBufferFn =
      TfLiteStatus (*)(const TfLiteTensor*, void*, size_t);

  TfLiteRuntime() = default;
  ~TfLiteRuntime() { Release(); }

  bool Load(const char* explicit_path) {
    if (handle_) {
      return true;
    }
    std::vector<std::string> candidates;
    if (explicit_path && explicit_path[0] != '\0') {
      candidates.emplace_back(explicit_path);
    } else {
#if defined(__APPLE__)
#if TARGET_OS_IPHONE
#if defined(__OBJC__)
      @autoreleasepool {
        NSString* bundlePath = [[NSBundle mainBundle] bundlePath];
        NSArray<NSString*>* roots = @[
          [[NSBundle mainBundle] privateFrameworksPath],
          bundlePath ? [bundlePath stringByAppendingPathComponent:@"Frameworks"] : nil,
          [[NSBundle mainBundle] resourcePath],
        ];
        for (NSString* root in roots) {
          if (![root length]) {
            continue;
          }
          NSString* frameworkBinary =
              [root stringByAppendingPathComponent:
                            @"TensorFlowLiteC.framework/TensorFlowLiteC"];
          candidates.emplace_back([frameworkBinary UTF8String]);
        }
      }
#endif  // defined(__OBJC__)
      candidates.emplace_back("TensorFlowLiteC.framework/TensorFlowLiteC");
      candidates.emplace_back("TensorFlowLiteC");
#endif  // TARGET_OS_IPHONE
      candidates.emplace_back("libtensorflowlite_c.dylib");
#elif defined(_WIN32)
      candidates.emplace_back("tensorflowlite_c.dll");
#else
      candidates.emplace_back("libtensorflowlite_c.so");
#endif
    }

    for (const std::string& candidate : candidates) {
#if defined(_WIN32)
      HMODULE module = LoadLibraryA(candidate.c_str());
      if (module) {
        handle_ = module;
        break;
      }
#else
      void* module = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_LOCAL);
      if (module) {
        handle_ = module;
        break;
      }
#endif
    }

    if (!handle_) {
#if defined(__APPLE__)
      // On Apple platforms the TensorFlowLiteC framework may be linked
      // statically into the final binary. In that case `dlopen` fails, but the
      // symbols are still available through RTLD_DEFAULT.
      handle_ = RTLD_DEFAULT;
      if (LoadSymbols()) {
        return true;
      }
      handle_ = nullptr;
#endif
      error_ = "TensorFlow Lite runtime library could not be loaded.";
      return false;
    }

    if (!LoadSymbols()) {
      Release();
      return false;
    }
    return true;
  }

  void Release() {
    if (handle_) {
#if defined(_WIN32)
      FreeLibrary(static_cast<HMODULE>(handle_));
#else
      if (handle_ != RTLD_DEFAULT) {
        dlclose(handle_);
      }
#endif
      handle_ = nullptr;
    }
    ModelCreateFromFile = nullptr;
    ModelDelete = nullptr;
    InterpreterOptionsCreate = nullptr;
    InterpreterOptionsDelete = nullptr;
    InterpreterOptionsSetThreads = nullptr;
    InterpreterCreate = nullptr;
    InterpreterDelete = nullptr;
    InterpreterAllocateTensors = nullptr;
    InterpreterInvoke = nullptr;
    InterpreterGetInputTensor = nullptr;
    InterpreterGetOutputTensor = nullptr;
    InterpreterGetInputTensorCount = nullptr;
    InterpreterGetOutputTensorCount = nullptr;
    TensorType = nullptr;
    TensorNumDims = nullptr;
    TensorDim = nullptr;
    TensorByteSize = nullptr;
    TensorData = nullptr;
    TensorCopyFromBuffer = nullptr;
    TensorCopyToBuffer = nullptr;
  }

  std::string error() const { return error_; }

  ModelCreateFromFileFn ModelCreateFromFile = nullptr;
  ModelDeleteFn ModelDelete = nullptr;
  InterpreterOptionsCreateFn InterpreterOptionsCreate = nullptr;
  InterpreterOptionsDeleteFn InterpreterOptionsDelete = nullptr;
  InterpreterOptionsSetThreadsFn InterpreterOptionsSetThreads = nullptr;
  InterpreterCreateFn InterpreterCreate = nullptr;
  InterpreterDeleteFn InterpreterDelete = nullptr;
  InterpreterAllocateTensorsFn InterpreterAllocateTensors = nullptr;
  InterpreterInvokeFn InterpreterInvoke = nullptr;
  InterpreterGetInputTensorFn InterpreterGetInputTensor = nullptr;
  InterpreterGetOutputTensorFn InterpreterGetOutputTensor = nullptr;
  InterpreterGetInputTensorCountFn InterpreterGetInputTensorCount = nullptr;
  InterpreterGetOutputTensorCountFn InterpreterGetOutputTensorCount = nullptr;
  TensorTypeFn TensorType = nullptr;
  TensorNumDimsFn TensorNumDims = nullptr;
  TensorDimFn TensorDim = nullptr;
  TensorByteSizeFn TensorByteSize = nullptr;
  TensorDataFn TensorData = nullptr;
  TensorCopyFromBufferFn TensorCopyFromBuffer = nullptr;
  TensorCopyToBufferFn TensorCopyToBuffer = nullptr;

 private:
  bool LoadSymbols() {
#define LOAD_SYMBOL(var, sym)                                                   \
  do {                                                                          \
    var = reinterpret_cast<decltype(var)>(ResolveSymbol(sym));                  \
    if (!var) {                                                                 \
      error_ = std::string("Unable to locate symbol: ") + sym;                  \
      return false;                                                             \
    }                                                                           \
  } while (false)

    LOAD_SYMBOL(ModelCreateFromFile, "TfLiteModelCreateFromFile");
    LOAD_SYMBOL(ModelDelete, "TfLiteModelDelete");
    LOAD_SYMBOL(InterpreterOptionsCreate, "TfLiteInterpreterOptionsCreate");
    LOAD_SYMBOL(InterpreterOptionsDelete, "TfLiteInterpreterOptionsDelete");
    LOAD_SYMBOL(InterpreterOptionsSetThreads,
                "TfLiteInterpreterOptionsSetNumThreads");
    LOAD_SYMBOL(InterpreterCreate, "TfLiteInterpreterCreate");
    LOAD_SYMBOL(InterpreterDelete, "TfLiteInterpreterDelete");
    LOAD_SYMBOL(InterpreterAllocateTensors, "TfLiteInterpreterAllocateTensors");
    LOAD_SYMBOL(InterpreterInvoke, "TfLiteInterpreterInvoke");
    LOAD_SYMBOL(InterpreterGetInputTensor, "TfLiteInterpreterGetInputTensor");
    LOAD_SYMBOL(InterpreterGetOutputTensor, "TfLiteInterpreterGetOutputTensor");
    LOAD_SYMBOL(InterpreterGetInputTensorCount,
                "TfLiteInterpreterGetInputTensorCount");
    LOAD_SYMBOL(InterpreterGetOutputTensorCount,
                "TfLiteInterpreterGetOutputTensorCount");
    LOAD_SYMBOL(TensorType, "TfLiteTensorType");
    LOAD_SYMBOL(TensorNumDims, "TfLiteTensorNumDims");
    LOAD_SYMBOL(TensorDim, "TfLiteTensorDim");
    LOAD_SYMBOL(TensorByteSize, "TfLiteTensorByteSize");
    LOAD_SYMBOL(TensorData, "TfLiteTensorData");
    LOAD_SYMBOL(TensorCopyFromBuffer, "TfLiteTensorCopyFromBuffer");
    LOAD_SYMBOL(TensorCopyToBuffer, "TfLiteTensorCopyToBuffer");
#undef LOAD_SYMBOL
    return true;
  }

  void* ResolveSymbol(const char* symbol) {
#if defined(_WIN32)
    return handle_ ? reinterpret_cast<void*>(
                         GetProcAddress(static_cast<HMODULE>(handle_), symbol))
                   : nullptr;
#else
    return handle_ ? dlsym(handle_, symbol) : nullptr;
#endif
  }

#if defined(_WIN32)
  HMODULE handle_ = nullptr;
#else
  void* handle_ = nullptr;
#endif
  std::string error_;
};

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

  MpFaceMeshResult* Process(const MpImage& image,
                            const MpNormalizedRect* override_rect) {
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

    MpNormalizedRect rect;
    if (override_rect) {
      rect = SanitizeRect(*override_rect);
      has_valid_rect_ = true;
    } else if (has_valid_rect_) {
      rect = roi_;
    } else {
      rect = DefaultRect();
    }

    if (!Preprocess(image, rect)) {
      return nullptr;
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

    MpFaceMeshResult* result = BuildResult(image, rect, score);
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

  void Shutdown() {
    interpreter_.reset();
    options_.reset();
    model_.reset();
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
    result->image_width = image.width;
    result->image_height = image.height;

    const RectInPixels roi = ToPixelRect(rect, image.width, image.height);
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
          Clamp(abs_x / static_cast<float>(image.width), -0.5f, 1.5f);
      landmark.y =
          Clamp(abs_y / static_cast<float>(image.height), -0.5f, 1.5f);
      landmark.z = abs_z / static_cast<float>(image.width);
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
    const MpNormalizedRect* override_rect) {
  if (!context) {
    SetGlobalError("Context is null.");
    return nullptr;
  }
  if (!image) {
    SetGlobalError("Image is null.");
    return nullptr;
  }
  return context->impl.Process(*image, override_rect);
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
