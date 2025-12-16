#ifndef TENSORFLOW_LITE_C_C_API_TYPES_H_
#define TENSORFLOW_LITE_C_C_API_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TFL_CAPI_EXPORT
#if defined(_WIN32)
#define TFL_CAPI_EXPORT __declspec(dllimport)
#else
#define TFL_CAPI_EXPORT __attribute__((visibility("default")))
#endif
#endif

typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteTensor TfLiteTensor;

typedef enum {
  kTfLiteOk = 0,
  kTfLiteError = 1,
  kTfLiteDelegateError = 2,
  kTfLiteApplicationError = 3,
} TfLiteStatus;

typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
  kTfLiteComplex128 = 12,
  kTfLiteUInt64 = 13,
  kTfLiteResource = 14,
  kTfLiteVariant = 15,
  kTfLiteUInt32 = 16,
  kTfLiteUInt16 = 17,
  kTfLiteInt4 = 18,
} TfLiteType;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_C_C_API_TYPES_H_
