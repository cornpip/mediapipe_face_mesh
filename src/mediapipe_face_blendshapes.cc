#include "mediapipe_face.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if defined(__OBJC__) && TARGET_OS_IPHONE
#import <Foundation/Foundation.h>
#endif
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
#define MP_BS_LOG_TAG "MediapipeBlendshapes"
#define MP_BS_LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, MP_BS_LOG_TAG, __VA_ARGS__)
#define MP_BS_LOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, MP_BS_LOG_TAG, __VA_ARGS__)
#else
#define MP_BS_LOGI(...) std::fprintf(stdout, "[INFO] " __VA_ARGS__)
#define MP_BS_LOGE(...) std::fprintf(stderr, "[ERROR] " __VA_ARGS__)
#endif

namespace {

constexpr int kBlendshapeCount = 52;
constexpr int kBlendshapeInputLandmarkCount = 146;
// The blendshapes model consumes a subset that reaches into the iris landmarks
// (468..477), so the source result must contain the full 478 landmarks.
constexpr int kMinRequiredLandmarks = 478;

// Subset of the 478 landmarks required by the blendshapes model, in the exact
// order expected by the model input. Mirrors MediaPipe's kLandmarksSubsetIdxs.
constexpr int kBlendshapeLandmarkSubset[kBlendshapeInputLandmarkCount] = {
    0,   1,   4,   5,   6,   7,   8,   10,  13,  14,  17,  21,  33,  37,  39,
    40,  46,  52,  53,  54,  55,  58,  61,  63,  65,  66,  67,  70,  78,  80,
    81,  82,  84,  87,  88,  91,  93,  95,  103, 105, 107, 109, 127, 132, 133,
    136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
    161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
    249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477};

class BlendshapesContext {
 public:
  BlendshapesContext() = default;
  ~BlendshapesContext() { Shutdown(); }

  bool Initialize(const std::string& model_path,
                  const MpBlendshapesCreateOptions* options) {
    threads_ = (options && options->threads > 0) ? options->threads : 2;

    const char* runtime_path = (options && options->tflite_library_path)
                                   ? options->tflite_library_path
                                   : nullptr;
    if (!runtime_.Load(runtime_path)) {
      SetError("Failed to load TensorFlow Lite runtime: " + runtime_.error());
      return false;
    }

    model_.reset(runtime_.ModelCreateFromFile(model_path.c_str()));
    if (!model_) {
      SetError("Unable to load blendshapes model file: " + model_path);
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
                              const char* name, MpDelegateType delegate_type) {
      if (!created) {
        return false;
      }
      delegate_.get_deleter().deleter = deleter;
      delegate_.reset(created);
      runtime_.InterpreterOptionsAddDelegate(
          options_.get(),
          reinterpret_cast<TfLiteOpaqueDelegate*>(delegate_.get()));
      active_delegate_ = delegate_type;
      MP_BS_LOGI("Blendshapes %s delegate enabled.\n", name);
      return true;
    };

    switch (delegate_choice) {
      case MP_DELEGATE_XNNPACK: {
        if (runtime_.InterpreterOptionsAddDelegate &&
            runtime_.XnnpackDelegateOptionsDefault &&
            runtime_.XnnpackDelegateCreate && runtime_.XnnpackDelegateDelete) {
          TfLiteXNNPackDelegateOptions xnnpack_options =
              runtime_.XnnpackDelegateOptionsDefault();
          xnnpack_options.num_threads = threads_;
          AttachDelegate(runtime_.XnnpackDelegateCreate(&xnnpack_options),
                         runtime_.XnnpackDelegateDelete, "XNNPACK",
                         MP_DELEGATE_XNNPACK);
        } else if (!allow_delegate_fallback) {
          SetError("XNNPACK delegate is unavailable for blendshapes model and "
                   "delegate fallback is disabled.");
          return false;
        }
        break;
      }
      case MP_DELEGATE_GPU_V2: {
        if (runtime_.InterpreterOptionsAddDelegate &&
            runtime_.GpuDelegateV2OptionsDefault &&
            runtime_.GpuDelegateV2Create && runtime_.GpuDelegateV2Delete) {
          TfLiteGpuDelegateOptionsV2 gpu_options =
              runtime_.GpuDelegateV2OptionsDefault();
          gpu_options.experimental_flags |=
              TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
          AttachDelegate(runtime_.GpuDelegateV2Create(&gpu_options),
                         runtime_.GpuDelegateV2Delete, "GPU V2",
                         MP_DELEGATE_GPU_V2);
        } else if (!allow_delegate_fallback) {
          SetError("GPU delegate (V2) is unavailable for blendshapes model and "
                   "delegate fallback is disabled.");
          return false;
        }
        break;
      }
      case MP_DELEGATE_CPU:
      default:
        break;
    }

    if (!allow_delegate_fallback && delegate_choice != MP_DELEGATE_CPU &&
        active_delegate_ != delegate_choice) {
      SetError("Failed to create requested delegate for blendshapes model "
               "because delegate fallback is disabled.");
      return false;
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
      MP_BS_LOGE(
          "Blendshapes interpreter creation with the requested delegate "
          "failed. Falling back to CPU.\n");
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
      SetError("Failed to create blendshapes interpreter.");
      return false;
    }
    if (!tensors_ready) {
      SetError("Blendshapes tensor allocation failed.");
      return false;
    }
    if (runtime_.InterpreterGetInputTensorCount(interpreter_.get()) < 1) {
      SetError("Blendshapes interpreter input tensor missing.");
      return false;
    }
    input_tensor_ = runtime_.InterpreterGetInputTensor(interpreter_.get(), 0);
    if (!input_tensor_ ||
        runtime_.TensorType(input_tensor_) != kTfLiteFloat32) {
      SetError("Blendshapes model input must be float32.");
      return false;
    }
    if (TensorElementCount(input_tensor_) !=
        static_cast<size_t>(kBlendshapeInputLandmarkCount * 2)) {
      SetError("Blendshapes model expects a 146x2 landmark input.");
      return false;
    }
    input_buffer_.resize(static_cast<size_t>(kBlendshapeInputLandmarkCount * 2));

    if (runtime_.InterpreterGetOutputTensorCount(interpreter_.get()) < 1) {
      SetError("Blendshapes model output missing.");
      return false;
    }
    output_tensor_ = runtime_.InterpreterGetOutputTensor(interpreter_.get(), 0);
    if (!output_tensor_ ||
        runtime_.TensorType(output_tensor_) != kTfLiteFloat32) {
      SetError("Blendshapes output tensor must be float32.");
      return false;
    }
    if (TensorElementCount(output_tensor_) !=
        static_cast<size_t>(kBlendshapeCount)) {
      SetError("Blendshapes model must output 52 coefficients.");
      return false;
    }
    output_buffer_.resize(static_cast<size_t>(kBlendshapeCount));
    MP_BS_LOGI("Blendshapes initialize success\n");
    return true;
  }

  MpBlendshapesResult* Process(const MpLandmark* landmarks,
                               int landmarks_count, int image_width,
                               int image_height) {
    if (!interpreter_) {
      SetError("Blendshapes interpreter is not initialized.");
      return nullptr;
    }
    if (!landmarks || landmarks_count < kMinRequiredLandmarks) {
      SetError("Blendshapes require at least 478 landmarks (enable iris).");
      return nullptr;
    }
    if (image_width <= 0 || image_height <= 0) {
      SetError("Invalid image size for blendshapes.");
      return nullptr;
    }

    const float width = static_cast<float>(image_width);
    const float height = static_cast<float>(image_height);
    for (int i = 0; i < kBlendshapeInputLandmarkCount; ++i) {
      const MpLandmark& landmark = landmarks[kBlendshapeLandmarkSubset[i]];
      input_buffer_[i * 2] = landmark.x * width;
      input_buffer_[i * 2 + 1] = landmark.y * height;
    }

    if (runtime_.TensorCopyFromBuffer(input_tensor_, input_buffer_.data(),
                                      input_buffer_.size() * sizeof(float)) !=
        kTfLiteOk) {
      SetError("Failed to copy blendshapes input buffer.");
      return nullptr;
    }
    if (runtime_.InterpreterInvoke(interpreter_.get()) != kTfLiteOk) {
      SetError("Blendshapes invocation failed.");
      return nullptr;
    }
    if (runtime_.TensorCopyToBuffer(output_tensor_, output_buffer_.data(),
                                    output_buffer_.size() * sizeof(float)) !=
        kTfLiteOk) {
      SetError("Unable to read blendshapes output.");
      return nullptr;
    }

    auto* result = new MpBlendshapesResult();
    result->scores = new float[kBlendshapeCount];
    result->scores_count = kBlendshapeCount;
    for (int i = 0; i < kBlendshapeCount; ++i) {
      result->scores[i] = output_buffer_[i];
    }
    return result;
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

  size_t TensorElementCount(const TfLiteTensor* tensor) const {
    int total = 1;
    const int dims = runtime_.TensorNumDims(tensor);
    for (int i = 0; i < dims; ++i) {
      total *= runtime_.TensorDim(tensor, i);
    }
    return static_cast<size_t>(total);
  }

  void SetError(const std::string& message) {
    last_error_ = message;
    MP_BS_LOGE("%s\n", message.c_str());
  }

  TfLiteRuntime runtime_;
  std::unique_ptr<TfLiteModel, TfLiteModelDeleter> model_{nullptr, {&runtime_}};
  std::unique_ptr<TfLiteInterpreterOptions, TfLiteOptionsDeleter> options_{
      nullptr, {&runtime_}};
  std::unique_ptr<TfLiteInterpreter, TfLiteInterpreterDeleter> interpreter_{
      nullptr, {&runtime_}};
  std::unique_ptr<TfLiteDelegate, TfLiteDelegateDeleter> delegate_{nullptr, {}};

  TfLiteTensor* input_tensor_ = nullptr;
  const TfLiteTensor* output_tensor_ = nullptr;

  int threads_ = 2;
  MpDelegateType active_delegate_ = MP_DELEGATE_CPU;

  std::vector<float> input_buffer_;
  std::vector<float> output_buffer_;
  std::string last_error_;
};

thread_local std::string g_bs_last_global_error;

void SetGlobalBlendshapesError(const std::string& message) {
  g_bs_last_global_error = message;
}

}  // namespace

struct MpBlendshapesContext {
  BlendshapesContext impl;
};

extern "C" {

FFI_PLUGIN_EXPORT MpBlendshapesContext* mp_blendshapes_create(
    const char* model_path, const MpBlendshapesCreateOptions* options) {
  if (!model_path) {
    SetGlobalBlendshapesError("Model path is null.");
    return nullptr;
  }
  auto* context = new MpBlendshapesContext();
  if (!context->impl.Initialize(model_path, options)) {
    SetGlobalBlendshapesError(context->impl.last_error());
    delete context;
    return nullptr;
  }
  return context;
}

FFI_PLUGIN_EXPORT void mp_blendshapes_destroy(MpBlendshapesContext* context) {
  delete context;
}

FFI_PLUGIN_EXPORT MpBlendshapesResult* mp_blendshapes_process(
    MpBlendshapesContext* context, const MpLandmark* landmarks,
    int32_t landmarks_count, int32_t image_width, int32_t image_height) {
  if (!context) {
    SetGlobalBlendshapesError("Context is null.");
    return nullptr;
  }
  return context->impl.Process(landmarks, landmarks_count, image_width,
                               image_height);
}

FFI_PLUGIN_EXPORT void mp_blendshapes_release_result(
    MpBlendshapesResult* result) {
  if (!result) {
    return;
  }
  delete[] result->scores;
  result->scores = nullptr;
  delete result;
}

FFI_PLUGIN_EXPORT const char* mp_blendshapes_last_error(
    const MpBlendshapesContext* context) {
  if (!context) {
    return nullptr;
  }
  return context->impl.last_error();
}

FFI_PLUGIN_EXPORT const char* mp_blendshapes_last_global_error(void) {
  return g_bs_last_global_error.c_str();
}

FFI_PLUGIN_EXPORT MpDelegateType mp_blendshapes_active_delegate(
    const MpBlendshapesContext* context) {
  if (!context) {
    return MP_DELEGATE_CPU;
  }
  return context->impl.active_delegate();
}

}  // extern "C"
