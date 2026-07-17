## 2.2.0

- align tracking-confidence semantics with the official graph: when a tracked
  face's presence score falls below `minTrackingConfidence`, the native tracked
  ROI is now dropped (and `FaceMeshInferencePipeline` re-acquires via the
  detector on the next frame) instead of freezing the last ROI while landmarks
  kept coming — previously a documented caveat when raising
  `minTrackingConfidence` above 0.5
  - the first tracked ROI seed is no longer smoothed against the initial
    full-frame rect, so the first tracked frame crops the actual face extent
- add `FaceMeshProcessor.isTracking`, backed by the new native
  `mp_face_mesh_is_tracking`: whether the internal ROI currently follows a face
- expose `minFacePresenceConfidence` on `FaceMeshProcessor.create` and
  `createForMultiFace` (the score below which a frame returns no landmarks;
  was always the native default 0.5 before) plus a matching getter
- add `FaceMeshProcessor.processRois` / `processNv21Rois`, backed by the new
  native `mp_face_mesh_process_rois` / `mp_face_mesh_process_rois_nv21`: one
  mesh inference per ROI on a single native frame upload, instead of copying
  the full frame into native memory once per face
  - `processMultiFace` / `processNv21MultiFace` and the
    `FaceMeshInferencePipeline` multi-face flow now route through it, so
    multi-face frames copy the frame across the FFI boundary once regardless
    of face count
  - pipeline multi-face acquisition now dedupes candidate detection ROIs
    against tracked faces and each other (same official IoU threshold) before
    one batched mesh call; per-face mesh results and track association are
    unchanged

## 2.1.0

- add `enableAttentionMesh` on `FaceMeshProcessor.create` and
  `FaceMeshProcessor.createForMultiFace` (default `false`, so existing setups are unchanged)
  - runs the official `face_landmark_with_attention` model, which refines lips, eyes, and
    irises in a single inference and outputs the 478-landmark layout directly, instead of the
    base 468-point mesh plus a separate iris pass
  - iris is always included when it is enabled, so it supersedes `enableIris` (the separate
    iris model is not loaded and `activeIrisDelegate` is null); `irisEnabled` stays true, so
    `FaceBlendshapesProcessor` keeps working
  - the attention outputs are merged into the existing index layout exactly as the official
    `LandmarksRefinementCalculator` does, so mesh/iris/blendshape/geometry consumers need no changes
  - XNNPACK does not support the model's custom ops; TFLite partitions the graph and runs those
    nodes on the reference CPU kernels, so `FaceMeshDelegate.xnnpack` accelerates only the rest
    of the model on the attention path
- add `FaceMeshProcessor.irisEnabled` (true whenever the processor returns 478 landmarks) and
  `FaceMeshProcessor.attentionMeshEnabled`
- rebuild the bundled TensorFlow Lite C runtimes (Android `arm64-v8a`/`x86_64`, iOS
  `TensorFlowLiteC.framework`) with the MediaPipe custom ops the attention model requires;
  the iOS framework also shrinks from ~47 MB to ~18 MB
- bundle `assets/models/face_landmark_with_attention.tflite` (~2.4 MB); it is only
  materialized to disk when `enableAttentionMesh` is set
- example: replace the `Iris` toggle with a mesh-mode selector (Base / Iris / Attention Mesh)

## 2.0.0

- **BREAKING**: `FaceMeshInferencePipeline` single-face flow now follows the official MediaPipe Face Mesh design — the detector runs only to (re)acquire a face, and tracked frames derive the mesh ROI from the previous frame's landmarks
  - improves mesh accuracy when the raw detection box is imprecise (e.g. a wide-open mouth no longer distorts the mesh) and roughly halves the per-frame compute on tracked frames
  - `FaceMeshInferenceResult.detectionResult` is now nullable; it is null on landmark-tracked frames where the detector did not run
  - on tracked frames `FaceMeshInferenceResult.selectedRoi` carries the tracked ROI used for mesh inference
  - since the detector is not consulted on tracked frames, `detectorRoi` and the detector ROI scale/shift overrides apply only to (re)acquisition frames
  - tracking resets automatically when input type, frame size, rotation, or mirroring changes; call `resetTracking()` when switching input sources that the pipeline cannot distinguish
  - add `FaceMeshInferencePipeline.isTracking` and `FaceMeshInferencePipeline.resetTracking()`, and `detectorRan` on both result types
  - add `FaceMeshProcessor.roiTrackingEnabled`; single-face landmark tracking requires a mesh processor created with `enableRoiTracking: true` (the default)
- **BREAKING**: multi-face pipeline methods (`processMultiFace`/`processNv21MultiFace`) also track by landmarks — each tracked face runs on an ROI derived from its previous frame's landmarks, and the detector runs only while fewer than `maxMeshFaces` faces are tracked, with IoU-gated association for newly detected faces (matching the official graph's `num_faces` behavior)
  - `FaceMeshMultiInferenceResult.detectionResult` is now nullable (null when all face slots were served by tracking) and results are exposed as `faces` — a `List<TrackedFaceMesh>` with a per-face `trackId` that stays stable while the face is tracked
  - multi-face tracking is managed in Dart with explicit per-face ROIs, so it works with `createForMultiFace` processors; the single- and multi-face flows share the native mesh state, so calling one resets the other's tracking and the other flow re-acquires via the detector on its next call
  - a tracked face is dropped — and its slot re-acquired via the detector — when its mesh presence score falls below `minTrackingConfidence` (now exposed as `FaceMeshProcessor.minTrackingConfidence`), matching the official tracking-confidence semantics
- add `FaceMeshResult.trackingRoi()` — the landmark-derived ROI a tracker uses for the next frame, matching the native formula; useful for custom pipelines
- add `FaceMeshPainter.scaleWithFace` — scales stroke widths and dot radii with each face's on-screen size (per face when drawing multiple results), so overlays keep their proportions as faces move closer or farther; the example enables it
- fix internal ROI tracking (`enableRoiTracking`) computing its square ROI in normalized space, which vertically stretched the ROI on portrait frames and degraded tracked landmark quality (present since 1.2.1); the ROI square and eye-line rotation are now computed in pixel space, matching the official graph
- fix ROI size sanitation clamping width and height independently, which could stretch very small or very large ROIs anisotropically; the size bounds are now applied with a single scale factor that preserves the ROI aspect
- fix a native result leak when copying a detector or mesh result to Dart throws; the native result is now released in a `finally`
- reduce internal ROI smoothing (alpha 0.8 → 0.5) so the tracked ROI follows fast face changes with less lag
- example: draw the tracked ROI overlay and show a `Tracking` status chip while the detector is skipped
- example: add a `Multi` toggle that switches to the multi-face tracking flow, rendering every tracked face's mesh and ROI with its `trackId` label
- example: remove the legacy ML Kit detector flow from the app; README links the v1.10.1 ML Kit integration for users who need an external detector example

### Migrating from 1.x

- `FaceMeshInferenceResult.detectionResult` and `FaceMeshMultiInferenceResult.detectionResult` are nullable — handle the null (tracked-frame) case; use `detectorRan` to tell the two frame types apart
- multi-face results moved from a `meshResults` constructor field to `faces` (`List<TrackedFaceMesh>`); reading `result.meshResults` still works via a getter
- to restore the previous every-frame detector behavior, pass `enableLandmarkTracking: false` to `FaceMeshInferencePipeline` (the nullable type change still applies)

## 1.10.1

- docs: update the example image
- add `.pubignore` to exclude README images from the published package (~7 MB smaller download)
- example: remove an unused asset

## 1.10.0

- add support for ARKit-style face blendshapes
  - `FaceBlendshapesProcessor`, a post-processor that turns a `FaceMeshResult` into the 52 blendshape coefficients (`Map<FaceBlendshape, double>`); requires a mesh created with `enableIris: true`
  - `FaceBlendshape` enum (52 categories, `neutral` at index 0)
  - bundles the `face_blendshapes.tflite` model (~0.9 MB)
- example: add a blendshapes-based expression detection demo

## 1.9.1

- patch release to re-trigger pub.dev analysis after 1.9.0 was stuck in pending state

## 1.9.0

- add `FaceMeshResult.estimateGeometry()` for 3D face geometry
  - `geometry.headPose` — head pose estimation (yaw, pitch, roll)
  - `geometry.distanceCm()` — centimeter distance between any two landmarks
  - `geometry.measurements` — preset bundle of common face measurements
  - accepts `verticalFovDegrees` for improved centimeter accuracy when the actual camera FOV is known (default: 63°)
- add `FaceMeshResult.distancePixels()` for 2D pixel distance between landmarks

## 1.8.1

- expose camera plane conversion logic as public `FaceMeshNv21Image` helper APIs

## 1.8.0

- add delegate fallback controls and active delegate diagnostics for face detector, face mesh, and optional iris model

## 1.7.1

- fix Android build configuration to align with Flutter 3.32.x (Dart 3.8.1) defaults
- set plugin `compileSdk` to 35 and `ndkVersion` to 26.3.11579264
- set example `minSdk` to 24

## 1.7.0

- add multi-face mesh inference APIs with `FaceMeshMultiInferenceResult`, `processMultiFace`, and `processNv21MultiFace`
- add reusable `FaceMeshPainter` and `FaceDetectionPainter` preview overlay painters as package files
- update the examples to use the public overlay painters
- update README docs for multi-face inference

## 1.6.0

- add `FaceMeshInferencePipeline` and `FaceMeshInferenceResult` for one-call unified detector and face mesh inference
- add `FaceMeshInferenceStreamProcessor` for stream-based unified inference
- update the MediaPipe detector example to use `FaceMeshInferencePipeline`
- update README usage docs for the unified inference API

## 1.5.0

- add bundled full-range dense and sparse face detector model support
- add `FaceDetectionModel` selection to `FaceDetectorProcessor.create`

## 1.4.1

- update docs

## 1.4.0

- add optional iris landmark output through `FaceMeshProcessor.create(enableIris: true)`

## 1.3.2

- add stream processing support to `FaceDetectorStreamProcessor` (`process` / `processNv21`)
- update README and example to cover FaceDetectorStreamProcessor stream inference

## 1.3.1

- lower the Dart SDK constraint to `^3.8.1`
- update the example camera adapter to handle multiple Android YUV layouts in Dart

## 1.3.0

- add `FaceDetectorProcessor` with bundled MediaPipe short-range face detection model
- support detector-driven ROI flow for face mesh inference and expose `FaceDetection` / `FaceDetectionResult`
- refactor example app to include both MediaPipe and ML Kit detection flows
- update README and change license to BSD 3-Clause

## 1.2.6

- add pub.dev topics and update package description

## 1.2.5

- rewrite example app: live camera demo using `google_mlkit_face_detection` + `FaceMeshStreamProcessor`
- render face mesh as polygon wireframe via `result.triangles` (`MpFaceMeshTriangle`)
- clean up README

## 1.2.4

- add MediaPipe face mesh triangulation topology and expose `FaceMeshResult.triangles`.
- sanitize the cache filename without `RegExp` to avoid the deprecation warning.

## 1.2.3

- update README to cover `FaceMeshResult` output fields, normalization rules, and ROI behavior
- add `toString()` overrides for core value classes (rect, box, image, landmark, result)

## 1.2.2

- add `enableRoiTracking` option in `FaceMeshProcessor.create` to control internal ROI tracking between frames

## 1.2.1

- add `enableRoiTracking` option in `FaceMeshProcessor.create` to control internal ROI tracking between frames

## 1.2.0

- improve README usage guidance and stream/camera documentation
- rename `FaceMeshStreamProcessor.processImages` to `FaceMeshStreamProcessor.process`
- adjust default `_boxScale` from 1.3 to 1.2 in `mediapipe_face_mesh.dart`

## 1.1.1

* document official LiteRT build instructions and expected binary locations in README

## 1.1.0

* replace bundled `tensorflow/lite` and `tensorflow/compiler` headers with upstream copies
* add runtime delegate selection (CPU / XNNPACK / GPU V2) and expose the option through the Dart API
* update README to reflect delegate support and document TensorFlow source folders

## 1.0.3

* run `dart format .` across the repo
* shorten `pubspec.yaml` description to satisfy length requirements

## 1.0.2

* add detailed plugin description and document key public APIs
* enable `public_member_api_docs` lint
* update README.md
* fix corrupted `example/assets/img.png` binary

## 1.0.1

* update docs

## 1.0.0

* Initial public release of the MediaPipe Face Mesh FFI plugin for Android and iOS
