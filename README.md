# mediapipe_face_mesh

Bundled files:
- TensorFlow Lite C runtime binaries for Android (`arm64-v8a`, `x86_64`) and iOS
- [MediaPipe TFLite model](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md)
  - face mesh
  - iris
  - short-range face detection
  - full-range dense and sparse face detection

<img src="./readme_img/2.png" alt="app_image_2" width="300"/> <img src="./readme_img/3.gif" alt="app_image_2" width="300"/>

## Supported Platforms

- Android(arm64-v8a, x86_64)
- iOS
- Dart SDK: `>=3.8.1 <4.0.0`

## Install

```bash
flutter pub add mediapipe_face_mesh
```

## Usage

### Create Face Detector Processor

```dart
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

final faceDetectorProcessor = await FaceDetectorProcessor.create(
  model: FaceDetectionModel.fullRange,
  delegate: FaceMeshDelegate.xnnpack,
  maxResults: 1,
  roiScaleY: 1.7,
  roiShiftY: -0.2,
);
```
`FaceDetectionModel` selects the bundled detector model:
`shortRange` is the default short-range BlazeFace model, `fullRange` is the
dense full-range model, and `fullRangeSparse` is the sparse full-range model.

ROI options adjust the detector-produced `expandedFaceRect`, which is passed to
face mesh while keeping the original frame unchanged.

### Create Face Mesh Processor

```dart
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

final faceMeshProcessor = await FaceMeshProcessor.create(
  delegate: FaceMeshDelegate.xnnpack,
  enableSmoothing: true,
  enableRoiTracking: true,
  enableIris: true, // default is false; true returns 478 landmarks with 10 iris points
);
```

When `enableIris` is enabled, Face Mesh runs an additional iris landmark pass
after the base 468-point face mesh result. The final result keeps the existing
Face Mesh index layout, updates the eye-region landmarks with more precise eye
contour coordinates, and appends 10 iris landmarks at indices `468..477`.

Delegate options:
- `FaceMeshDelegate.cpu` (default)
- `FaceMeshDelegate.xnnpack`
- `FaceMeshDelegate.gpuV2`

If the requested delegate is unavailable or fails to initialize, the native
runtime falls back to CPU inference.

### Input Formats

The package supports two image input types:

- `FaceMeshNv21Image`
  Use this for Android camera frames in NV21 layout.
- `FaceMeshImage`
  Use this for RGBA or BGRA buffers. This is used for iOS camera frames.

### Stream Inference

Use stream inference when processing continuous camera frames. Stream processors
take a Stream of frames and return a Stream of results.

```dart
final pipeline = FaceMeshInferencePipeline(
  detector: faceDetectorProcessor,
  mesh: faceMeshProcessor,
);
final inferenceStreamProcessor = FaceMeshInferenceStreamProcessor(pipeline);
final frameController = StreamController<FaceMeshNv21Image>();
bool _isBusy = false;

inferenceStreamProcessor
    .processNv21(
      frameController.stream,
      runMeshResolver: (_) => _isMeshActive,
      rotationDegrees: rotationDegrees,
    )
    .listen(_handleInferenceResult, onError: onError);

void _handleInferenceResult(FaceMeshInferenceResult result) {
  _isBusy = false;
  onDetections(result.detectionResult);
  onMeshResult(result.meshResult);
}

void onCameraFrame(FaceMeshNv21Image frame) {
  if (_isBusy) return;
  _isBusy = true;
  frameController.add(frame);
}
```

Use `runMesh: false` when an entire stream should run detector-only. Use
`runMeshResolver` when mesh execution should be decided per frame, such as a UI
toggle that can change while the stream is active.

For BGRA / RGBA input, use `process(...)` instead of `processNv21(...)`.

### Single Inference

Use single-frame inference in one call without a stream processor.

```dart
final pipeline = FaceMeshInferencePipeline(
  detector: faceDetectorProcessor,
  mesh: faceMeshProcessor,
);

final result = pipeline.processNv21(
  nv21Image,
  rotationDegrees: rotationDegrees,
);

final meshResult = result.meshResult;
if (meshResult != null) {
  onResult(meshResult);
}
```

For detector-only, set `runMesh: false`.

```dart
final result = pipeline.processNv21(
  nv21Image,
  runMesh: false,
  rotationDegrees: rotationDegrees,
);
```

### Multi-Face Inference

Multi-face mesh inference runs face detection once, then runs mesh inference for
each selected detector ROI. Use `maxResults` on the detector to control how many
faces are detected, and `maxMeshFaces` on the multi-face mesh call to control
how many mesh inferences are run.

Create the mesh processor with tracking and smoothing disabled so state from one
face ROI does not affect the next face ROI.

`createForMultiFace(...)` is a convenience factory equivalent to
`FaceMeshProcessor.create(..., enableSmoothing: false, enableRoiTracking: false)`.

```dart
final faceMeshProcessor = await FaceMeshProcessor.createForMultiFace(
  delegate: FaceMeshDelegate.xnnpack,
  enableIris: true,
);
```

```dart
final faceDetectorProcessor = await FaceDetectorProcessor.create(
  delegate: FaceMeshDelegate.xnnpack,
  maxResults: 4,
);
final pipeline = FaceMeshInferencePipeline(
  detector: faceDetectorProcessor,
  mesh: faceMeshProcessor,
);
final inferenceStreamProcessor = FaceMeshInferenceStreamProcessor(pipeline);

inferenceStreamProcessor
    .processNv21MultiFace(
      frameController.stream,
      maxMeshFaces: 2,
      runMeshResolver: (_) => _isMeshActive,
      rotationDegrees: rotationDegrees,
    )
    .listen(_handleMultiInferenceResult, onError: onError);

void _handleMultiInferenceResult(FaceMeshMultiInferenceResult result) {
  onDetections(result.detectionResult);
  onMeshResults(result.meshResults);
}
```

For BGRA / RGBA input, use `processMultiFace(...)` instead of
`processNv21MultiFace(...)`.

For single-frame multi-face inference, call the pipeline directly.

```dart
final FaceMeshMultiInferenceResult result = pipeline.processNv21MultiFace(
  nv21Image,
  maxMeshFaces: 4,
  rotationDegrees: rotationDegrees,
);

final FaceDetectionResult detections = result.detectionResult;
final List<FaceMeshResult> meshResults = result.meshResults;
```

### Close Resource

Explicitly calling close() when the processors are no longer needed is recommended.

```dart
faceDetectorProcessor.close();
faceMeshProcessor.close();
```

### Notes

The examples in this README use the v1.6.0+ unified inference API.

`FaceMeshInferenceStreamProcessor` emits one combined result after detector and
mesh inference complete, so detection boxes and mesh landmarks are updated
together.

If you need detector boxes to update independently from slower mesh inference,
use `FaceDetectorStreamProcessor` and `FaceMeshStreamProcessor` separately; see
the [v1.5.0](https://github.com/cornpip/mediapipe_face_mesh/tree/v1.5.0) README and example app
for a two-stage stream pattern.

These separated stream processors are still available in v1.6.0 and later.

## Example app

The example app in `example/` provides two flows:

A. MediaPipe Face Detector + MediaPipe Face Mesh  
B. ML Kit Face Detector + MediaPipe Face Mesh

`B` depends on the `google_mlkit_face_detection` package for face detection.

## Primary API

- `FaceMeshInferencePipeline`
  Runs face detection and face mesh inference in one call for single-frame use.
- `FaceMeshInferenceStreamProcessor`
  Wraps `FaceMeshInferencePipeline` in an `async*` generator — accepts a
  `Stream` of frames and yields a `Stream` of high-level inference results.
- `FaceMeshInferenceResult`
  Contains detector output, selected detection, selected box/ROI, ROI
  availability, and mesh output.
- `FaceMeshMultiInferenceResult`
  Contains detector output and all mesh outputs produced from detector ROIs.
- `FaceDetectorProcessor`
  Runs the bundled MediaPipe short-range, full-range dense, or full-range sparse
  face detector and returns face boxes, scores, and rotation-aware ROI values
  such as `expandedFaceRect`.
- `FaceMeshProcessor`
  Runs face mesh inference and returns normalized 3D landmarks, mesh triangles,
  the detected face rect, score, and input image size.
- `FaceMeshNv21Image`
  Input wrapper for Android NV21 camera frames.
- `FaceMeshImage`
  Input wrapper for RGBA or BGRA pixel buffers.
- `NormalizedRect`
  Rotation-aware normalized ROI used to restrict face mesh inference.
- `FaceMeshBox`
  Pixel-space bounding box that can be converted into an ROI internally.
- `FaceMeshResult`
  Result object containing `landmarks`, `triangles`, `rect`, `score`,
  `imageWidth`, and `imageHeight`. Face mesh returns 468 landmarks by default,
  or 478 landmarks when `enableIris` is enabled. Pixel-space helpers such as
  `landmarkAsOffset(...)` and `landmarksAsOffsets(...)` support rotation and
  horizontal mirror mapping for preview overlays.
- `FaceMeshPainter`
  Optional `CustomPainter` for drawing one or more face mesh results
- `FaceDetectionPainter`
  Optional `CustomPainter` for drawing `FaceDetection` boxes
