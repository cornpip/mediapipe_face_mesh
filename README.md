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

final faceDetector = await FaceDetectorProcessor.create(
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

ROI options adjust the detector-produced `expandedFaceRect` used for later face
mesh inference.

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
  Use this for RGBA or BGRA buffers. This is the used for iOS camera frames.

### Stream Inference

Use stream inference when processing continuous camera frames; use single inference for one-shot images.  
Both `StreamProcessor` classes take a Stream of frames and return a Stream of results.

```dart
final streamProcessor = FaceMeshStreamProcessor(faceMeshProcessor);
final frameController = StreamController<FaceMeshNv21Image>();
bool _isBusy = false;

void onCameraFrame(FaceMeshNv21Image frame) {
  if (_isBusy) return;          // drop frame — previous inference still running
  _isBusy = true;
  frameController.add(frame);   // returns immediately; camera session unblocked
}

streamProcessor
    .processNv21(frameController.stream, rotationDegrees: rotationDegrees)
    .listen((result) {
      _isBusy = false;
      onResult(result);          // update overlay; camera was never blocked
    }, onError: onError);
```

Full two-stage (detector → mesh) example:

```dart
final detectorStreamProcessor = FaceDetectorStreamProcessor(faceDetectorProcessor);
final streamProcessor = FaceMeshStreamProcessor(faceMeshProcessor);
NormalizedRect? latestRoi;
bool _isDetectorBusy = false;
bool _isMeshBusy = false;
final detectorFrameController = StreamController<FaceMeshNv21Image>();
final meshFrameController = StreamController<FaceMeshNv21Image>();

detectorStreamProcessor
    .processNv21(
      detectorFrameController.stream,
      rotationDegrees: rotationDegrees,
    )
    .listen((detectionResult) {
      _isDetectorBusy = false;
      latestRoi = detectionResult.primaryDetection?.expandedFaceRect;
      if (latestRoi != null && !_isMeshBusy) {
        _isMeshBusy = true;
        meshFrameController.add(lastFrame!);
      }
    }, onError: onError);

streamProcessor
    .processNv21(
      meshFrameController.stream,
      roiResolver: (_) => latestRoi,
      rotationDegrees: rotationDegrees,
    )
    .listen((result) {
      _isMeshBusy = false;
      onResult(result);
    }, onError: onError);

void onCameraFrame(FaceMeshNv21Image frame) {
  lastFrame = frame;
  if (!_isDetectorBusy) {
    _isDetectorBusy = true;
    detectorFrameController.add(frame);
  }
}
```

For BGRA / RGBA input, use `process(...)` instead of `processNv21(...)`.

### Single Inference

```dart
final detectionResult = faceDetectorProcessor.processNv21(
  nv21Image,
  rotationDegrees: rotationDegrees,
);
final detection = detectionResult.primaryDetection;

if (detection != null) {
  final result = faceMeshProcessor.processNv21(
    nv21Image,
    roi: detection.expandedFaceRect,
    rotationDegrees: rotationDegrees,
  );
}
```

### ROI Inputs

Face Mesh accepts ROI input in two ways.

For single-frame inference, use `roi` or `box`.
For stream inference, the same distinction applies through `roiResolver` and
`boxResolver`.

- `roi`
  Pass the final `NormalizedRect` directly.
  Use this when you already have a rotation-aware ROI such as `expandedFaceRect`.
- `box`
  Pass a `FaceMeshBox`, which is converted internally into a normalized ROI.
  This path applies clamping, `boxScale`, and `boxMakeSquare`, and produces an
  axis-aligned ROI (`rotation == 0`).

If both `roi` and `box` are provided, an `ArgumentError` is thrown.

### Close Resource

Explicitly calling close() when the processors are no longer needed is recommended.

```dart
faceDetectorProcessor.close();
faceMeshProcessor.close();
```

## Example app

The example app in `example/` provides two flows:

A. MediaPipe Face Detector + MediaPipe Face Mesh  
B. ML Kit Face Detector + MediaPipe Face Mesh

`B` depends on the `google_mlkit_face_detection` package for face detection.

## Primary API

- `FaceDetectorProcessor`
  Runs the bundled MediaPipe short-range, full-range dense, or full-range sparse
  face detector and returns face boxes, scores, and rotation-aware ROI values
  such as `expandedFaceRect`.
- `FaceDetectorStreamProcessor`
  Wraps `FaceDetectorProcessor` in an `async*` generator — accepts a `Stream` of frames and yields a `Stream` of results.
- `FaceMeshProcessor`
  Runs face mesh inference and returns normalized 3D landmarks, mesh triangles,
  the detected face rect, score, and input image size.
- `FaceMeshStreamProcessor`
  Wraps `FaceMeshProcessor` in an `async*` generator — accepts a `Stream` of frames and yields a `Stream` of results.
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
