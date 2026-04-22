# mediapipe_face_mesh

MediaPipe Face Mesh for Flutter.

Bundled files:
- MediaPipe Face Mesh TFLite model
- MediaPipe short-range face detection model
- TensorFlow Lite C runtime binaries for Android (`arm64-v8a`, `x86_64`) and iOS

Reference: [MediaPipe TFLite models](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md)

## Install

```bash
flutter pub add mediapipe_face_mesh
```

## Usage

### Create Face Detector Processor

```dart
final faceDetector = await FaceDetectorProcessor.create(
  delegate: FaceMeshDelegate.xnnpack,
  maxResults: 1,
  roiScaleY: 1.7,
  roiShiftY: -0.2,
);
```

ROI options adjust the `expandedFaceRect` region used for face mesh
inference.


### Create Face Mesh Processor
```dart
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

final faceMeshProcessor = await FaceMeshProcessor.create(
  delegate: FaceMeshDelegate.xnnpack,
);
```
delegates options:

- `FaceMeshDelegate.cpu` (default)
- `FaceMeshDelegate.xnnpack`
- `FaceMeshDelegate.gpuV2`

If the requested delegate is unavailable or fails to initialize, back to CPU inference.

### Stream Inference

```dart
final streamProcessor = FaceMeshStreamProcessor(faceMeshProcessor);
NormalizedRect? latestRoi;
final frameController = StreamController<FaceMeshNv21Image>();

streamProcessor
    .processNv21(
      frameController.stream,
      roiResolver: (_) => latestRoi,
      rotationDegrees: rotationDegrees,
    )
    .listen(onResult, onError: onError);

void onCameraFrame(FaceMeshNv21Image frame) {
  final detectionResult = faceDetector.processNv21(
    frame,
    rotationDegrees: rotationDegrees,
  );
  latestRoi = detectionResult.primaryDetection?.expandedFaceRect;

  if (latestRoi != null) {
    frameController.add(frame);
  }
}
```

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

**Face Mesh accepts ROI input in two ways.**

For single-frame inference, use `roi` or `box`.
For stream inference, the same distinction applies through `roiResolver` and `boxResolver`.

- `roi`
  pass the final `NormalizedRect` directly.
  Use this when you already have a rotation-aware ROI such as `expandedFaceRect`.
- `box`
  pass a `FaceMeshBox`, which is converted internally into a normalized ROI.
  This path applies clamping, `boxScale`, and `boxMakeSquare`, and produces an axis-aligned ROI (`rotation == 0`).

If both `roi` and `box` are provided, an `ArgumentError` is thrown


## Example

The example included in this package provides two example flows:

A. MediaPipe Face Detector + MediaPipe Face Mesh  
B. ML Kit Face Detector + MediaPipe Face Mesh

`B` depends on the `google_mlkit_face_detection` flutter package for face detection.

<img src="./readme_img/2.png" alt="app_image_2" width="300"/>
<img src="./readme_img/3.gif" alt="app_image_2" width="300"/>
