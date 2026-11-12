#include "mediapipe_face.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "tflite_c_api.h"

#if defined(__ANDROID__)
#include <android/log.h>
#define MP_IN_LOG_TAG "MediapipeInterpreter"
#define MP_IN_LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, MP_IN_LOG_TAG, __VA_ARGS__)
#define MP_IN_LOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, MP_IN_LOG_TAG, __VA_ARGS__)
#else
#define MP_IN_LOGI(...) std::fprintf(stdout, "[INFO] " __VA_ARGS__)
#define MP_IN_LOGE(...) std::fprintf(stderr, "[ERROR] " __VA_ARGS__)
#endif

namespace {

class InterpreterContext {
 public:
  InterpreterContext() = default;
  ~InterpreterContext() { Shutdown(); }

  bool InitializeFromFile(const std::string& model_path,
                          const MpInterpreterCreateOptions* options) {
    model_.reset(TfLiteModelCreateFromFile(model_path.c_str()));
    if (!model_) {
      SetError("Unable to load model file: " + model_path);
      return false;
    }
    return BuildInterpreter(options);
  }

  bool InitializeFromBuffer(const uint8_t* data, uint64_t size,
                            const MpInterpreterCreateOptions* options) {
    // TfLiteModelCreate does not copy: the bytes must outlive the model.
    model_data_.assign(data, data + static_cast<size_t>(size));
    model_.reset(TfLiteModelCreate(model_data_.data(), model_data_.size()));
    if (!model_) {
      SetError("Unable to load model from the given buffer.");
      return false;
    }
    return BuildInterpreter(options);
  }

  int32_t input_count() const {
    return interpreter_
               ? TfLiteInterpreterGetInputTensorCount(interpreter_.get())
               : 0;
  }

  int32_t output_count() const {
    return interpreter_
               ? TfLiteInterpreterGetOutputTensorCount(interpreter_.get())
               : 0;
  }

  bool InputInfo(int32_t index, MpTensorInfo* info) {
    return FillInfo(InputTensor(index), info);
  }

  bool OutputInfo(int32_t index, MpTensorInfo* info) {
    return FillInfo(OutputTensor(index), info);
  }

  void* InputData(int32_t index) {
    TfLiteTensor* tensor = InputTensor(index);
    return tensor ? TfLiteTensorData(tensor) : nullptr;
  }

  const void* OutputData(int32_t index) {
    const TfLiteTensor* tensor = OutputTensor(index);
    return tensor ? TfLiteTensorData(tensor) : nullptr;
  }

  bool Invoke() {
    if (!interpreter_) {
      SetError("Interpreter is not initialized.");
      return false;
    }
    if (TfLiteInterpreterInvoke(interpreter_.get()) != kTfLiteOk) {
      SetError("Interpreter invocation failed.");
      return false;
    }
    return true;
  }

  const char* last_error() const { return last_error_.c_str(); }

  // Records an error raised outside the class, e.g. a C++ exception caught
  // at the FFI boundary.
  void RecordError(const std::string& message) { SetError(message); }

  MpDelegateType active_delegate() const { return active_delegate_; }

 private:
  struct TfLiteModelDeleter {
    void operator()(TfLiteModel* model) const {
      if (model) {
        TfLiteModelDelete(model);
      }
    }
  };

  struct TfLiteOptionsDeleter {
    void operator()(TfLiteInterpreterOptions* options) const {
      if (options) {
        TfLiteInterpreterOptionsDelete(options);
      }
    }
  };

  struct TfLiteInterpreterDeleter {
    void operator()(TfLiteInterpreter* interpreter) const {
      if (interpreter) {
        TfLiteInterpreterDelete(interpreter);
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

  bool ResetOptions() {
    options_.reset(TfLiteInterpreterOptionsCreate());
    if (!options_) {
      SetError("Failed to allocate interpreter options.");
      return false;
    }
    TfLiteInterpreterOptionsSetNumThreads(options_.get(), threads_);
    return true;
  }

  bool BuildInterpreter(const MpInterpreterCreateOptions* options) {
    threads_ = (options && options->threads > 0) ? options->threads : 2;
    if (!ResetOptions()) {
      return false;
    }

    const MpDelegateType delegate_choice =
        options ? static_cast<MpDelegateType>(options->delegate)
                : MP_DELEGATE_CPU;
    const bool allow_delegate_fallback =
        !options || options->disable_delegate_fallback == 0;
    active_delegate_ = MP_DELEGATE_CPU;

    if (delegate_choice == MP_DELEGATE_XNNPACK) {
      TfLiteXNNPackDelegateOptions xnnpack_options =
          TfLiteXNNPackDelegateOptionsDefault();
      xnnpack_options.num_threads = threads_;
      TfLiteDelegate* created = TfLiteXNNPackDelegateCreate(&xnnpack_options);
      if (created) {
        delegate_.get_deleter().deleter = TfLiteXNNPackDelegateDelete;
        delegate_.reset(created);
        TfLiteInterpreterOptionsAddDelegate(
            options_.get(),
            reinterpret_cast<TfLiteOpaqueDelegate*>(delegate_.get()));
        active_delegate_ = MP_DELEGATE_XNNPACK;
        MP_IN_LOGI("Interpreter XNNPACK delegate enabled.\n");
      }
    }

    if (!allow_delegate_fallback && delegate_choice != MP_DELEGATE_CPU &&
        active_delegate_ != delegate_choice) {
      SetError("Failed to create the requested delegate because delegate "
               "fallback is disabled.");
      return false;
    }

    interpreter_.reset(TfLiteInterpreterCreate(model_.get(), options_.get()));
    bool tensors_ready =
        interpreter_ &&
        TfLiteInterpreterAllocateTensors(interpreter_.get()) == kTfLiteOk;
    // A delegate can also fail after it is attached, while the interpreter
    // builds or allocates tensors. Honor delegate fallback for that stage
    // too.
    if (!tensors_ready && active_delegate_ != MP_DELEGATE_CPU &&
        allow_delegate_fallback) {
      MP_IN_LOGE("Interpreter creation with the requested delegate failed. "
                 "Falling back to CPU.\n");
      interpreter_.reset();
      delegate_.reset();
      if (!ResetOptions()) {
        return false;
      }
      active_delegate_ = MP_DELEGATE_CPU;
      interpreter_.reset(
          TfLiteInterpreterCreate(model_.get(), options_.get()));
      tensors_ready =
          interpreter_ &&
          TfLiteInterpreterAllocateTensors(interpreter_.get()) == kTfLiteOk;
    }
    if (!interpreter_) {
      SetError("Failed to create the interpreter. The model may use an "
               "operator the bundled runtime does not include.");
      return false;
    }
    if (!tensors_ready) {
      SetError("Tensor allocation failed.");
      return false;
    }
    return true;
  }

  TfLiteTensor* InputTensor(int32_t index) {
    if (!interpreter_) {
      SetError("Interpreter is not initialized.");
      return nullptr;
    }
    if (index < 0 || index >= input_count()) {
      SetError("Input tensor index out of range: " + std::to_string(index));
      return nullptr;
    }
    return TfLiteInterpreterGetInputTensor(interpreter_.get(), index);
  }

  const TfLiteTensor* OutputTensor(int32_t index) {
    if (!interpreter_) {
      SetError("Interpreter is not initialized.");
      return nullptr;
    }
    if (index < 0 || index >= output_count()) {
      SetError("Output tensor index out of range: " + std::to_string(index));
      return nullptr;
    }
    return TfLiteInterpreterGetOutputTensor(interpreter_.get(), index);
  }

  bool FillInfo(const TfLiteTensor* tensor, MpTensorInfo* info) {
    if (!tensor) {
      return false;
    }
    if (!info) {
      SetError("Tensor info output is null.");
      return false;
    }
    const int32_t dims_count = TfLiteTensorNumDims(tensor);
    if (dims_count < 0 || dims_count > MP_TENSOR_MAX_DIMS) {
      SetError("Unsupported tensor rank: " + std::to_string(dims_count));
      return false;
    }
    info->name = TfLiteTensorName(tensor);
    info->type = static_cast<int32_t>(TfLiteTensorType(tensor));
    info->dims_count = dims_count;
    for (int32_t i = 0; i < MP_TENSOR_MAX_DIMS; ++i) {
      info->dims[i] = i < dims_count ? TfLiteTensorDim(tensor, i) : 0;
    }
    info->byte_size = static_cast<uint64_t>(TfLiteTensorByteSize(tensor));
    return true;
  }

  void Shutdown() {
    interpreter_.reset();
    options_.reset();
    model_.reset();
    delegate_.reset();
    model_data_.clear();
  }

  void SetError(const std::string& message) {
    last_error_ = message;
    MP_IN_LOGE("%s\n", message.c_str());
  }

  // Backs model_ when it was created from a buffer.
  std::vector<uint8_t> model_data_;
  std::unique_ptr<TfLiteModel, TfLiteModelDeleter> model_;
  std::unique_ptr<TfLiteInterpreterOptions, TfLiteOptionsDeleter> options_;
  std::unique_ptr<TfLiteInterpreter, TfLiteInterpreterDeleter> interpreter_;
  std::unique_ptr<TfLiteDelegate, TfLiteDelegateDeleter> delegate_{nullptr, {}};

  int threads_ = 2;
  MpDelegateType active_delegate_ = MP_DELEGATE_CPU;
  std::string last_error_;
};

thread_local std::string g_in_last_global_error;

void SetGlobalInterpreterError(const std::string& message) {
  g_in_last_global_error = message;
}

}  // namespace

struct MpInterpreterContext {
  InterpreterContext impl;
};

namespace {

// No C++ exception may unwind across the FFI boundary into Dart (the process
// would abort), so exports run their body through this guard, which converts
// an escaped exception (std::bad_alloc included) into [failure] with the
// error recorded on the context, when there is one, and globally.
template <typename Fn, typename Result>
Result GuardedNative(MpInterpreterContext* context, Result failure,
                     Fn&& body) {
  try {
    return body();
  } catch (const std::exception& e) {
    const std::string message =
        std::string("Unhandled native exception: ") + e.what();
    if (context) {
      context->impl.RecordError(message);
    }
    SetGlobalInterpreterError(message);
    return failure;
  } catch (...) {
    const std::string message = "Unhandled native exception.";
    if (context) {
      context->impl.RecordError(message);
    }
    SetGlobalInterpreterError(message);
    return failure;
  }
}

}  // namespace

extern "C" {

FFI_PLUGIN_EXPORT MpInterpreterContext* mp_interpreter_create_from_file(
    const char* model_path, const MpInterpreterCreateOptions* options) {
  return GuardedNative(
      nullptr, static_cast<MpInterpreterContext*>(nullptr),
      [&]() -> MpInterpreterContext* {
        if (!model_path) {
          SetGlobalInterpreterError("Model path is null.");
          return nullptr;
        }
        auto context = std::make_unique<MpInterpreterContext>();
        if (!context->impl.InitializeFromFile(model_path, options)) {
          SetGlobalInterpreterError(context->impl.last_error());
          return nullptr;
        }
        return context.release();
      });
}

FFI_PLUGIN_EXPORT MpInterpreterContext* mp_interpreter_create_from_buffer(
    const uint8_t* model_data, uint64_t model_size,
    const MpInterpreterCreateOptions* options) {
  return GuardedNative(
      nullptr, static_cast<MpInterpreterContext*>(nullptr),
      [&]() -> MpInterpreterContext* {
        if (!model_data || model_size == 0) {
          SetGlobalInterpreterError("Model buffer is empty.");
          return nullptr;
        }
        auto context = std::make_unique<MpInterpreterContext>();
        if (!context->impl.InitializeFromBuffer(model_data, model_size,
                                                options)) {
          SetGlobalInterpreterError(context->impl.last_error());
          return nullptr;
        }
        return context.release();
      });
}

FFI_PLUGIN_EXPORT void mp_interpreter_destroy(MpInterpreterContext* context) {
  try {
    delete context;
  } catch (...) {
  }
}

FFI_PLUGIN_EXPORT int32_t mp_interpreter_input_count(
    const MpInterpreterContext* context) {
  return context ? context->impl.input_count() : 0;
}

FFI_PLUGIN_EXPORT int32_t mp_interpreter_output_count(
    const MpInterpreterContext* context) {
  return context ? context->impl.output_count() : 0;
}

FFI_PLUGIN_EXPORT uint8_t mp_interpreter_input_info(
    MpInterpreterContext* context, int32_t index, MpTensorInfo* info) {
  return GuardedNative(context, static_cast<uint8_t>(0), [&]() -> uint8_t {
    return context && context->impl.InputInfo(index, info) ? 1 : 0;
  });
}

FFI_PLUGIN_EXPORT uint8_t mp_interpreter_output_info(
    MpInterpreterContext* context, int32_t index, MpTensorInfo* info) {
  return GuardedNative(context, static_cast<uint8_t>(0), [&]() -> uint8_t {
    return context && context->impl.OutputInfo(index, info) ? 1 : 0;
  });
}

FFI_PLUGIN_EXPORT void* mp_interpreter_input_data(
    MpInterpreterContext* context, int32_t index) {
  return GuardedNative(context, static_cast<void*>(nullptr), [&]() -> void* {
    return context ? context->impl.InputData(index) : nullptr;
  });
}

FFI_PLUGIN_EXPORT const void* mp_interpreter_output_data(
    MpInterpreterContext* context, int32_t index) {
  return GuardedNative(context, static_cast<const void*>(nullptr),
                       [&]() -> const void* {
                         return context ? context->impl.OutputData(index)
                                        : nullptr;
                       });
}

FFI_PLUGIN_EXPORT uint8_t mp_interpreter_invoke(MpInterpreterContext* context) {
  return GuardedNative(context, static_cast<uint8_t>(0), [&]() -> uint8_t {
    return context && context->impl.Invoke() ? 1 : 0;
  });
}

FFI_PLUGIN_EXPORT const char* mp_interpreter_last_error(
    const MpInterpreterContext* context) {
  if (!context) {
    return nullptr;
  }
  return context->impl.last_error();
}

FFI_PLUGIN_EXPORT const char* mp_interpreter_last_global_error(void) {
  return g_in_last_global_error.c_str();
}

FFI_PLUGIN_EXPORT MpDelegateType mp_interpreter_active_delegate(
    const MpInterpreterContext* context) {
  if (!context) {
    return MP_DELEGATE_CPU;
  }
  return context->impl.active_delegate();
}

}  // extern "C"
