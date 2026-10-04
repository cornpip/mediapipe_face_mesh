#ifndef TFLITE_C_API_H_
#define TFLITE_C_API_H_

// TensorFlow Lite C API headers for the bundled runtime, which every platform
// links directly.
#if defined(__APPLE__)
// Apple builds use the headers bundled in TensorFlowLiteC.xcframework; the
// other platforms use src/include.
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#else
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif

#endif  // TFLITE_C_API_H_
