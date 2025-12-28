#ifndef TFLITE_RUNTIME_H_
#define TFLITE_RUNTIME_H_

#include <string>
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

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

// Lightweight wrapper that loads the TensorFlow Lite C API at runtime.
class TfLiteRuntime {
 public:
  using ModelCreateFromFileFn = TfLiteModel* (*)(const char*);
  using ModelDeleteFn = void (*)(TfLiteModel*);
  using InterpreterOptionsCreateFn = TfLiteInterpreterOptions* (*)();
  using InterpreterOptionsDeleteFn = void (*)(TfLiteInterpreterOptions*);
  using InterpreterOptionsSetThreadsFn =
      void (*)(TfLiteInterpreterOptions*, int32_t);
  using InterpreterCreateFn =
      TfLiteInterpreter* (*)(const TfLiteModel*, const TfLiteInterpreterOptions*);
  using InterpreterDeleteFn = void (*)(TfLiteInterpreter*);
  using InterpreterAllocateTensorsFn = TfLiteStatus (*)(TfLiteInterpreter*);
  using InterpreterInvokeFn = TfLiteStatus (*)(TfLiteInterpreter*);
  using InterpreterGetInputTensorFn =
      TfLiteTensor* (*)(const TfLiteInterpreter*, int32_t);
  using InterpreterGetOutputTensorFn =
      const TfLiteTensor* (*)(const TfLiteInterpreter*, int32_t);
  using InterpreterGetInputTensorCountFn = int32_t (*)(const TfLiteInterpreter*);
  using InterpreterGetOutputTensorCountFn = int32_t (*)(const TfLiteInterpreter*);
  using TensorTypeFn = TfLiteType (*)(const TfLiteTensor*);
  using TensorNumDimsFn = int32_t (*)(const TfLiteTensor*);
  using TensorDimFn = int32_t (*)(const TfLiteTensor*, int32_t);
  using TensorByteSizeFn = size_t (*)(const TfLiteTensor*);
  using TensorDataFn = void* (*)(const TfLiteTensor*);
  using TensorCopyFromBufferFn = TfLiteStatus (*)(TfLiteTensor*, const void*, size_t);
  using TensorCopyToBufferFn =
      TfLiteStatus (*)(const TfLiteTensor*, void*, size_t);
  using InterpreterOptionsAddDelegateFn =
      void (*)(TfLiteInterpreterOptions*, TfLiteOpaqueDelegate*);
  using XnnpackDelegateCreateFn =
      TfLiteDelegate* (*)(const TfLiteXNNPackDelegateOptions*);
  using XnnpackDelegateDeleteFn = void (*)(TfLiteDelegate*);
  using XnnpackDelegateOptionsDefaultFn = TfLiteXNNPackDelegateOptions (*)();
  using GpuDelegateV2CreateFn =
      TfLiteDelegate* (*)(const TfLiteGpuDelegateOptionsV2*);
  using GpuDelegateV2DeleteFn = void (*)(TfLiteDelegate*);
  using GpuDelegateV2OptionsDefaultFn = TfLiteGpuDelegateOptionsV2 (*)();

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
    InterpreterOptionsAddDelegate = nullptr;
    XnnpackDelegateCreate = nullptr;
    XnnpackDelegateDelete = nullptr;
    XnnpackDelegateOptionsDefault = nullptr;
    GpuDelegateV2Create = nullptr;
    GpuDelegateV2Delete = nullptr;
    GpuDelegateV2OptionsDefault = nullptr;
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
  InterpreterOptionsAddDelegateFn InterpreterOptionsAddDelegate = nullptr;
  XnnpackDelegateCreateFn XnnpackDelegateCreate = nullptr;
  XnnpackDelegateDeleteFn XnnpackDelegateDelete = nullptr;
  XnnpackDelegateOptionsDefaultFn XnnpackDelegateOptionsDefault = nullptr;
  GpuDelegateV2CreateFn GpuDelegateV2Create = nullptr;
  GpuDelegateV2DeleteFn GpuDelegateV2Delete = nullptr;
  GpuDelegateV2OptionsDefaultFn GpuDelegateV2OptionsDefault = nullptr;

 private:
  bool LoadSymbols() {
    ModelCreateFromFile =
        reinterpret_cast<ModelCreateFromFileFn>(LoadSymbol("TfLiteModelCreateFromFile"));
    ModelDelete = reinterpret_cast<ModelDeleteFn>(LoadSymbol("TfLiteModelDelete"));
    InterpreterOptionsCreate = reinterpret_cast<InterpreterOptionsCreateFn>(
        LoadSymbol("TfLiteInterpreterOptionsCreate"));
    InterpreterOptionsDelete = reinterpret_cast<InterpreterOptionsDeleteFn>(
        LoadSymbol("TfLiteInterpreterOptionsDelete"));
    InterpreterOptionsSetThreads = reinterpret_cast<InterpreterOptionsSetThreadsFn>(
        LoadSymbol("TfLiteInterpreterOptionsSetNumThreads"));
    InterpreterOptionsAddDelegate =
        reinterpret_cast<InterpreterOptionsAddDelegateFn>(
            LoadSymbol("TfLiteInterpreterOptionsAddDelegate"));
    InterpreterCreate = reinterpret_cast<InterpreterCreateFn>(
        LoadSymbol("TfLiteInterpreterCreate"));
    InterpreterDelete = reinterpret_cast<InterpreterDeleteFn>(
        LoadSymbol("TfLiteInterpreterDelete"));
    InterpreterAllocateTensors = reinterpret_cast<InterpreterAllocateTensorsFn>(
        LoadSymbol("TfLiteInterpreterAllocateTensors"));
    InterpreterInvoke =
        reinterpret_cast<InterpreterInvokeFn>(LoadSymbol("TfLiteInterpreterInvoke"));
    InterpreterGetInputTensor = reinterpret_cast<InterpreterGetInputTensorFn>(
        LoadSymbol("TfLiteInterpreterGetInputTensor"));
    InterpreterGetOutputTensor = reinterpret_cast<InterpreterGetOutputTensorFn>(
        LoadSymbol("TfLiteInterpreterGetOutputTensor"));
    InterpreterGetInputTensorCount = reinterpret_cast<InterpreterGetInputTensorCountFn>(
        LoadSymbol("TfLiteInterpreterGetInputTensorCount"));
    InterpreterGetOutputTensorCount = reinterpret_cast<InterpreterGetOutputTensorCountFn>(
        LoadSymbol("TfLiteInterpreterGetOutputTensorCount"));
    TensorType = reinterpret_cast<TensorTypeFn>(LoadSymbol("TfLiteTensorType"));
    TensorNumDims = reinterpret_cast<TensorNumDimsFn>(LoadSymbol("TfLiteTensorNumDims"));
    TensorDim = reinterpret_cast<TensorDimFn>(LoadSymbol("TfLiteTensorDim"));
    TensorByteSize =
        reinterpret_cast<TensorByteSizeFn>(LoadSymbol("TfLiteTensorByteSize"));
    TensorData = reinterpret_cast<TensorDataFn>(LoadSymbol("TfLiteTensorData"));
    TensorCopyFromBuffer = reinterpret_cast<TensorCopyFromBufferFn>(
        LoadSymbol("TfLiteTensorCopyFromBuffer"));
    TensorCopyToBuffer =
        reinterpret_cast<TensorCopyToBufferFn>(LoadSymbol("TfLiteTensorCopyToBuffer"));
    XnnpackDelegateCreate = reinterpret_cast<XnnpackDelegateCreateFn>(
        LoadSymbolOptional("TfLiteXNNPackDelegateCreate"));
    XnnpackDelegateDelete = reinterpret_cast<XnnpackDelegateDeleteFn>(
        LoadSymbolOptional("TfLiteXNNPackDelegateDelete"));
    XnnpackDelegateOptionsDefault = reinterpret_cast<XnnpackDelegateOptionsDefaultFn>(
        LoadSymbolOptional("TfLiteXNNPackDelegateOptionsDefault"));
    GpuDelegateV2Create = reinterpret_cast<GpuDelegateV2CreateFn>(
        LoadSymbolOptional("TfLiteGpuDelegateV2Create"));
    GpuDelegateV2Delete = reinterpret_cast<GpuDelegateV2DeleteFn>(
        LoadSymbolOptional("TfLiteGpuDelegateV2Delete"));
    GpuDelegateV2OptionsDefault = reinterpret_cast<GpuDelegateV2OptionsDefaultFn>(
        LoadSymbolOptional("TfLiteGpuDelegateOptionsV2Default"));

    if (!ModelCreateFromFile || !ModelDelete || !InterpreterOptionsCreate ||
        !InterpreterOptionsDelete || !InterpreterOptionsSetThreads ||
        !InterpreterOptionsAddDelegate || !InterpreterCreate || !InterpreterDelete ||
        !InterpreterAllocateTensors || !InterpreterInvoke ||
        !InterpreterGetInputTensor || !InterpreterGetOutputTensor ||
        !InterpreterGetInputTensorCount || !InterpreterGetOutputTensorCount || !TensorType ||
        !TensorNumDims || !TensorDim || !TensorByteSize || !TensorData ||
        !TensorCopyFromBuffer || !TensorCopyToBuffer) {
      error_ = "Missing required TensorFlow Lite symbols.";
      return false;
    }
    return true;
  }

  void* LoadSymbol(const char* symbol) {
    if (!handle_) {
      return nullptr;
    }
#if defined(_WIN32)
    void* address = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), symbol));
#else
    void* address = dlsym(handle_, symbol);
#endif
    if (!address) {
      error_ = std::string("Failed to load symbol: ") + symbol;
    }
    return address;
  }

  void* LoadSymbolOptional(const char* symbol) {
    if (!handle_) {
      return nullptr;
    }
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), symbol));
#else
    return dlsym(handle_, symbol);
#endif
  }

  void* handle_ = nullptr;
  std::string error_;
};

#endif  // TFLITE_RUNTIME_H_
