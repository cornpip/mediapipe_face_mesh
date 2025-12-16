#ifndef MEDIAPIPE_FACE_MESH_H_
#define MEDIAPIPE_FACE_MESH_H_

#include <stdint.h>

#if _WIN32
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FFI_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MpFaceMeshContext MpFaceMeshContext;
typedef struct MpFaceDetectionContext MpFaceDetectionContext;

typedef enum {
  MP_PIXEL_FORMAT_RGBA = 0,
  MP_PIXEL_FORMAT_BGRA = 1,
} MpPixelFormat;

typedef struct {
  const uint8_t* data;
  int32_t width;
  int32_t height;
  int32_t bytes_per_row;
  MpPixelFormat format;
} MpImage;

typedef struct {
  float x_center;
  float y_center;
  float width;
  float height;
  float rotation;
} MpNormalizedRect;

typedef struct {
  float x;
  float y;
  float z;
} MpLandmark;

typedef struct {
  MpLandmark* landmarks;
  int32_t landmarks_count;
  MpNormalizedRect rect;
  float score;
  int32_t image_width;
  int32_t image_height;
} MpFaceMeshResult;

typedef struct {
  const char* tflite_library_path;
  int32_t threads;
  float min_detection_confidence;
  float min_tracking_confidence;
  uint8_t enable_smoothing;
} MpFaceMeshCreateOptions;

typedef struct {
  const char* tflite_library_path;
  int32_t threads;
  float score_threshold;
  float nms_threshold;
  int32_t max_detections;
} MpFaceDetectionCreateOptions;

typedef struct {
  float x_center;
  float y_center;
  float width;
  float height;
} MpDetectionBox;

typedef struct {
  MpDetectionBox box;
  float score;
  // Optional: BlazeFace short-range outputs 6 keypoints.
  float keypoints[12];  // x0,y0,x1,y1,...
  int32_t keypoints_count;
} MpDetection;

typedef struct {
  MpDetection* detections;
  int32_t count;
  int32_t image_width;
  int32_t image_height;
} MpFaceDetectionResult;

FFI_PLUGIN_EXPORT MpFaceMeshContext* mp_face_mesh_create(
    const char* model_path, const MpFaceMeshCreateOptions* options);

FFI_PLUGIN_EXPORT void mp_face_mesh_destroy(MpFaceMeshContext* context);

FFI_PLUGIN_EXPORT MpFaceMeshResult* mp_face_mesh_process(
    MpFaceMeshContext* context,
    const MpImage* image,
    const MpNormalizedRect* override_rect);

FFI_PLUGIN_EXPORT void mp_face_mesh_release_result(MpFaceMeshResult* result);

FFI_PLUGIN_EXPORT const char* mp_face_mesh_last_error(
    const MpFaceMeshContext* context);

FFI_PLUGIN_EXPORT const char* mp_face_mesh_last_global_error(void);

// Face detection (short-range) APIs.
FFI_PLUGIN_EXPORT MpFaceDetectionContext* mp_face_detection_create(
    const char* model_path, const MpFaceDetectionCreateOptions* options);

FFI_PLUGIN_EXPORT void mp_face_detection_destroy(
    MpFaceDetectionContext* context);

FFI_PLUGIN_EXPORT MpFaceDetectionResult* mp_face_detection_process(
    MpFaceDetectionContext* context, const MpImage* image);

FFI_PLUGIN_EXPORT void mp_face_detection_release_result(
    MpFaceDetectionResult* result);

FFI_PLUGIN_EXPORT const char* mp_face_detection_last_error(
    const MpFaceDetectionContext* context);

FFI_PLUGIN_EXPORT const char* mp_face_detection_last_global_error(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_FACE_MESH_H_
