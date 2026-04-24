# mediapipe_face_mesh

MediaPipe Face Mesh for Flutter with bundled native(c/c++) runtimes for Android and iOS.

Bundled files:
- MediaPipe Face Mesh TFLite model
- MediaPipe short-range face detection model
- TensorFlow Lite C runtime binaries for Android (`arm64-v8a`, `x86_64`) and iOS

Reference: [MediaPipe TFLite models](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md)

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
  delegate: FaceMeshDelegate.xnnpack,
  maxResults: 1,
  roiScaleY: 1.7,
  roiShiftY: -0.2,
);
```

ROI options adjust the detector-produced `expandedFaceRect` used for later face
mesh inference.

### Create Face Mesh Processor

```dart
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

final faceMeshProcessor = await FaceMeshProcessor.create(
  delegate: FaceMeshDelegate.xnnpack,
  enableSmoothing: true,
  enableRoiTracking: true,
);
```

Delegate options:
- `FaceMeshDelegate.cpu` (default)
- `FaceMeshDelegate.xnnpack`
- `FaceMeshDelegate.gpuV2`

If the requested delegate is unavailable or fails to initialize, the native
runtime falls back to CPU inference.

Useful processor options:
- `enableSmoothing`
  Reduces landmark jitter across frames.
- `enableRoiTracking`
  Reuses internal ROI tracking when later `process` or `processNv21` calls omit
  `roi` and `box`.
- `mirrorHorizontal`
  Use this in `process`, `processNv21`, or stream processing when the preview is
  mirrored.

### Input Formats

The package supports two image input types:

- `FaceMeshNv21Image`
  Use this for Android camera frames in NV21 layout.
- `FaceMeshImage`
  Use this for RGBA or BGRA buffers. This is the used for iOS camera frames.

### Stream Inference

Both `FaceDetectorStreamProcessor` and `FaceMeshStreamProcessor` process frames
through Dart `async*` generators. Inference is decoupled from the camera frame
callback: the camera session pushes frames into a `StreamController` and returns
immediately, while inference runs asynchronously on the next event loop turn.
This means **slow inference or a heavy `CustomPaint` repaint never stalls the
camera session** — the preview continues at full frame rate and frames that
arrive while the previous inference is still running are simply dropped.

The recommended pattern is to guard each push with a busy flag so you never
queue more than one frame at a time:

```dart
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
final detectorStreamProcessor = FaceDetectorStreamProcessor(faceDetector);
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
final detectionResult = faceDetector.processNv21(
  nv21Image,
  rotationDegrees: rotationDegrees,
);
final detection = detectionResult.primaryDetection;

if (detection != null) {
  final box = detection.toBox(
    imageWidth: detectionResult.imageWidth,
    imageHeight: detectionResult.imageHeight,
  );

  final result = faceMeshProcessor.processNv21(
    nv21Image,
    box: box,
    boxScale: 1.2,
    boxMakeSquare: true,
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
faceDetector.close();
faceMeshProcessor.close();
```

## Example

The example included in this package provides two flows:

A. MediaPipe Face Detector + MediaPipe Face Mesh
B. ML Kit Face Detector + MediaPipe Face Mesh

`B` depends on the `google_mlkit_face_detection` package for face detection.

<img src="./readme_img/2.png" alt="app_image_2" width="300"/>
<img src="./readme_img/3.gif" alt="app_image_2" width="300"/>

## Primary API

- `FaceDetectorProcessor`
  Runs the bundled MediaPipe short-range face detector and returns face boxes,
  scores, and rotation-aware ROI values such as `expandedFaceRect`.
- `FaceDetectorStreamProcessor`
  Wraps `FaceDetectorProcessor` as a non-blocking async stream. Camera frames
  are pushed via a `StreamController` and inference runs in an `async*` generator,
  so the camera session is never stalled by slow inference.
- `FaceMeshProcessor`
  Runs face mesh inference and returns normalized 3D landmarks, mesh triangles,
  the detected face rect, score, and input image size.
- `FaceMeshStreamProcessor`
  Wraps `FaceMeshProcessor` as a non-blocking async stream. Same decoupled
  design as `FaceDetectorStreamProcessor` — heavy `CustomPaint` repaints do not
  block the camera session.
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
  `imageWidth`, and `imageHeight`. Pixel-space helpers such as
  `landmarkAsOffset(...)` and `landmarksAsOffsets(...)` support rotation and
  horizontal mirror mapping for preview overlays.
