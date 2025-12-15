#ifndef TENSORFLOW_LITE_C_C_API_H_
#define TENSORFLOW_LITE_C_C_API_H_

#include "tensorflow/lite/c/c_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model
TFL_CAPI_EXPORT TfLiteModel* TfLiteModelCreateFromFile(const char* model_path);
TFL_CAPI_EXPORT void TfLiteModelDelete(TfLiteModel* model);

// Interpreter options
TFL_CAPI_EXPORT TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate(void);
TFL_CAPI_EXPORT void TfLiteInterpreterOptionsDelete(
    TfLiteInterpreterOptions* options);
TFL_CAPI_EXPORT void TfLiteInterpreterOptionsSetNumThreads(
    TfLiteInterpreterOptions* options, int32_t num_threads);

// Interpreter lifecycle
TFL_CAPI_EXPORT TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model, const TfLiteInterpreterOptions* optional_options);
TFL_CAPI_EXPORT void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter);
TFL_CAPI_EXPORT TfLiteStatus TfLiteInterpreterAllocateTensors(
    TfLiteInterpreter* interpreter);
TFL_CAPI_EXPORT TfLiteStatus TfLiteInterpreterInvoke(
    TfLiteInterpreter* interpreter);

// Interpreter I/O
TFL_CAPI_EXPORT TfLiteTensor* TfLiteInterpreterGetInputTensor(
    TfLiteInterpreter* interpreter, int32_t input_index);
TFL_CAPI_EXPORT const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t output_index);
TFL_CAPI_EXPORT int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter);
TFL_CAPI_EXPORT int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter);

// Tensor
TFL_CAPI_EXPORT TfLiteType TfLiteTensorType(const TfLiteTensor* tensor);
TFL_CAPI_EXPORT int TfLiteTensorNumDims(const TfLiteTensor* tensor);
TFL_CAPI_EXPORT int TfLiteTensorDim(const TfLiteTensor* tensor, int dim_index);
TFL_CAPI_EXPORT size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);
TFL_CAPI_EXPORT void* TfLiteTensorData(const TfLiteTensor* tensor);
TFL_CAPI_EXPORT TfLiteStatus TfLiteTensorCopyFromBuffer(
    TfLiteTensor* tensor, const void* input_data, size_t input_data_size);
TFL_CAPI_EXPORT TfLiteStatus TfLiteTensorCopyToBuffer(
    const TfLiteTensor* tensor, void* output_data, size_t output_data_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_C_C_API_H_
