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
  // One entry per input ROI, in input order. Entries whose face presence
  // score fell below the threshold have landmarks_count == 0, matching
  // mp_face_mesh_process.
  MpFaceMeshResult* results;
  int32_t results_count;
} MpFaceMeshMultiResult;

typedef struct {
  // Vertical field of view of the virtual camera in degrees.
  float vertical_fov_degrees;
  // Near and far clip planes of the virtual frustum in the same units as the
  // output metric landmarks (centimeters).
  float near_plane;
  float far_plane;
  // Non-zero when input normalized landmarks use a top-left image origin.
  // This matches FaceMeshResult landmarks returned by this plugin.
  uint8_t origin_top_left;
} MpFaceGeometryOptions;

typedef struct {
  // Landmarks reprojected into centimeter-scale metric space.
  MpLandmark* metric_landmarks;
  int32_t metric_landmarks_count;
  // Row-major 4x4 similarity transform from canonical face space to camera
  // space. Encodes rotation, translation, and uniform scale.
  float pose_transform_matrix[16];
  float yaw_degrees;
  float pitch_degrees;
  float roll_degrees;
  // Uniform scale factor from the Procrustes fit: ratio of the observed face
  // size to the canonical model size in centimeters.
  float scale;
} MpFaceGeometryResult;

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
  // When non-zero, [model_path] points to the unified
  // `face_landmark_with_attention` model, which refines lips, eyes, and irises
  // in a single inference and outputs 478 landmarks directly. Requires a
  // libtensorflowlite_c that registers the MediaPipe custom ops. When set, the
  // separate iris pass ([enable_iris]/[iris_model_path]) is not used; iris is
  // always included in the result.
  uint8_t enable_attention_mesh;
  // When non-zero, fail creation instead of falling back to CPU if the
  // requested delegate is unavailable or cannot be created.
  uint8_t disable_delegate_fallback;
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
  // When non-zero, fail creation instead of falling back to CPU if the
  // requested delegate is unavailable or cannot be created.
  uint8_t disable_delegate_fallback;
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

// Runs one mesh inference per ROI on a single frame upload, avoiding the
// per-face frame copy of calling mp_face_mesh_process once per face. Returns
// null and sets the context error if any ROI fails. Release the result with
// mp_face_mesh_release_multi_result.
FFI_PLUGIN_EXPORT MpFaceMeshMultiResult* mp_face_mesh_process_rois(
    MpFaceMeshContext* context,
    const MpImage* image,
    const MpNormalizedRect* rois,
    int32_t rois_count,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal);

FFI_PLUGIN_EXPORT MpFaceMeshMultiResult* mp_face_mesh_process_rois_nv21(
    MpFaceMeshContext* context,
    const MpNv21Image* image,
    const MpNormalizedRect* rois,
    int32_t rois_count,
    int32_t rotation_degrees,
    uint8_t mirror_horizontal);

FFI_PLUGIN_EXPORT void mp_face_mesh_release_result(MpFaceMeshResult* result);

FFI_PLUGIN_EXPORT void mp_face_mesh_release_multi_result(
    MpFaceMeshMultiResult* result);

// Non-zero while the internal tracked ROI follows a face — it was seeded
// from landmarks and has not been dropped by a face-presence or
// tracking-confidence failure, or an input-geometry change. Always zero when
// the context was created with enable_roi_tracking off.
FFI_PLUGIN_EXPORT uint8_t mp_face_mesh_is_tracking(
    const MpFaceMeshContext* context);

FFI_PLUGIN_EXPORT const char* mp_face_mesh_last_error(
    const MpFaceMeshContext* context);

FFI_PLUGIN_EXPORT const char* mp_face_mesh_last_global_error(void);

FFI_PLUGIN_EXPORT MpDelegateType mp_face_mesh_active_delegate(
    const MpFaceMeshContext* context);

FFI_PLUGIN_EXPORT MpDelegateType mp_face_mesh_active_iris_delegate(
    const MpFaceMeshContext* context);

FFI_PLUGIN_EXPORT MpFaceGeometryResult* mp_face_geometry_estimate(
    const MpLandmark* landmarks,
    int32_t landmarks_count,
    int32_t image_width,
    int32_t image_height,
    const MpFaceGeometryOptions* options);

FFI_PLUGIN_EXPORT void mp_face_geometry_release_result(
    MpFaceGeometryResult* result);

FFI_PLUGIN_EXPORT const char* mp_face_geometry_last_error(void);

// ---- Face blendshapes ------------------------------------------------------
// Post-processing module that turns face landmarks into 52 ARKit-style
// blendshape coefficients. Runs as its own context, separate from the face
// mesh context, on the landmarks it is given.

typedef struct MpBlendshapesContext MpBlendshapesContext;

typedef struct {
  const char* tflite_library_path;
  int32_t threads;
  MpDelegateType delegate;
  // When non-zero, fail creation instead of falling back to CPU if the
  // requested delegate is unavailable or cannot be created.
  uint8_t disable_delegate_fallback;
} MpBlendshapesCreateOptions;

typedef struct {
  // 52 blendshape coefficients in [0, 1], ordered to match the MediaPipe face
  // blendshapes model (index 0 is `_neutral`).
  float* scores;
  int32_t scores_count;
} MpBlendshapesResult;

FFI_PLUGIN_EXPORT MpBlendshapesContext* mp_blendshapes_create(
    const char* model_path, const MpBlendshapesCreateOptions* options);

FFI_PLUGIN_EXPORT void mp_blendshapes_destroy(MpBlendshapesContext* context);

// Runs the blendshapes model on the provided face landmarks. Requires at least
// 478 landmarks (iris). [image_width]/[image_height] are the dimensions the
// normalized landmarks are expressed against. Returns null on error.
FFI_PLUGIN_EXPORT MpBlendshapesResult* mp_blendshapes_process(
    MpBlendshapesContext* context,
    const MpLandmark* landmarks,
    int32_t landmarks_count,
    int32_t image_width,
    int32_t image_height);

FFI_PLUGIN_EXPORT void mp_blendshapes_release_result(
    MpBlendshapesResult* result);

FFI_PLUGIN_EXPORT const char* mp_blendshapes_last_error(
    const MpBlendshapesContext* context);

FFI_PLUGIN_EXPORT const char* mp_blendshapes_last_global_error(void);

FFI_PLUGIN_EXPORT MpDelegateType mp_blendshapes_active_delegate(
    const MpBlendshapesContext* context);

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

FFI_PLUGIN_EXPORT MpDelegateType mp_face_detector_active_delegate(
    const MpFaceDetectorContext* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_FACE_MESH_H_
