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
typedef struct MpFaceDetectorContext MpFaceDetectorContext;

typedef enum {
  MP_PIXEL_FORMAT_RGBA = 0,
  MP_PIXEL_FORMAT_BGRA = 1,
} MpPixelFormat;

typedef enum {
  MP_DELEGATE_CPU = 0,
  MP_DELEGATE_XNNPACK = 1,
  MP_DELEGATE_GPU_V2 = 2,
} MpDelegateType;

typedef struct {
  const uint8_t* data;
  int32_t width;
  int32_t height;
  int32_t bytes_per_row;
  MpPixelFormat format;
} MpImage;

// Android camera NV21 input (Y plane + interleaved VU plane).
typedef struct {
  const uint8_t* y;
  const uint8_t* vu;
  int32_t width;
  int32_t height;
  int32_t y_bytes_per_row;
  int32_t vu_bytes_per_row;
} MpNv21Image;

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
  const char* iris_model_path;
  int32_t threads;
  float min_detection_confidence;
  float min_tracking_confidence;
  float min_face_presence_confidence;
  MpDelegateType delegate;
  uint8_t enable_smoothing;
  uint8_t enable_roi_tracking;
  uint8_t enable_iris;
} MpFaceMeshCreateOptions;

typedef struct {
  float left;
  float top;
  float right;
  float bottom;
  float score;
  float keypoints[12];
  MpNormalizedRect face_rect;
  MpNormalizedRect expanded_face_rect;
} MpDetection;

typedef struct {
  MpDetection* detections;
  int32_t detections_count;
  int32_t image_width;
  int32_t image_height;
} MpFaceDetectorResult;

typedef struct {
  const char* tflite_library_path;
  int32_t threads;
  float min_detection_confidence;
  float min_suppression_threshold;
  int32_t max_results;
  MpDelegateType delegate;
} MpFaceDetectorCreateOptions;

typedef struct {
  float scale_x;
  float scale_y;
  float shift_x;
  float shift_y;
} MpRoiTransformOptions;

FFI_PLUGIN_EXPORT MpFaceMeshContext* mp_face_mesh_create(
    const char* model_path, const MpFaceMeshCreateOptions* options);

FFI_PLUGIN_EXPORT void mp_face_mesh_destroy(MpFaceMeshContext* context);

FFI_PLUGIN_EXPORT MpFaceMeshResult* mp_face_mesh_process(
    MpFaceMeshContext* context,
    const MpImage* image,
    const MpNormalizedRect* override_rect,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal);

FFI_PLUGIN_EXPORT MpFaceMeshResult* mp_face_mesh_process_nv21(
    MpFaceMeshContext* context,
    const MpNv21Image* image,
    const MpNormalizedRect* override_rect,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal);

FFI_PLUGIN_EXPORT void mp_face_mesh_release_result(MpFaceMeshResult* result);

FFI_PLUGIN_EXPORT const char* mp_face_mesh_last_error(
    const MpFaceMeshContext* context);

FFI_PLUGIN_EXPORT const char* mp_face_mesh_last_global_error(void);

FFI_PLUGIN_EXPORT MpFaceDetectorContext* mp_face_detector_create(
    const char* model_path, const MpFaceDetectorCreateOptions* options);

FFI_PLUGIN_EXPORT void mp_face_detector_destroy(MpFaceDetectorContext* context);

FFI_PLUGIN_EXPORT MpFaceDetectorResult* mp_face_detector_process(
    MpFaceDetectorContext* context,
    const MpImage* image,
    const MpNormalizedRect* override_rect,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal,
    const MpRoiTransformOptions* roi_transform);

FFI_PLUGIN_EXPORT MpFaceDetectorResult* mp_face_detector_process_nv21(
    MpFaceDetectorContext* context,
    const MpNv21Image* image,
    const MpNormalizedRect* override_rect,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal,
    const MpRoiTransformOptions* roi_transform);

FFI_PLUGIN_EXPORT void mp_face_detector_release_result(
    MpFaceDetectorResult* result);

FFI_PLUGIN_EXPORT const char* mp_face_detector_last_error(
    const MpFaceDetectorContext* context);

FFI_PLUGIN_EXPORT const char* mp_face_detector_last_global_error(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_FACE_MESH_H_
