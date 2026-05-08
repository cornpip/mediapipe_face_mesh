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
