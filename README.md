# mediapipe_face_mesh

Face detection and a 478-landmark face mesh pipeline, on device, in a few
milliseconds per frame. Models and the TensorFlow Lite runtime ship inside
the package.

<img src="./readme_img/22.png" alt="app_image_2" width="300"/> <img src="./readme_img/33.png" alt="app_image_2" width="300"/>

## Supported Platforms

| platform | requirement |
| --- | --- |
| Android | minSdk 24 (arm64-v8a, x86_64) |
| iOS | 13.0+ |
| Windows | x64 |

Requires Dart `>=3.8.1 <4.0.0` and Flutter `>=3.32.0`.

## Performance

Same device (Dimensity 9400, Android 16), same inputs. One call runs
detection plus the full face mesh in both packages.

| | mediapipe_face_mesh | google_mlkit_face_mesh_detection 0.5.0 |
| --- |---------------------| --- |
| single image, per call | 3~5 ms              | ~44 ms |
| streaming, per frame | 1~3 ms              | ~50 ms |

In streaming, mediapipe_face_mesh tracks the face between frames, while
ML Kit re-runs detection every frame (it has no tracking mode).
Single-image latency varies with device thermal state. Streaming is the
stable metric. Method, full matrix, and caveats in
[doc/BENCHMARKS.md](doc/BENCHMARKS.md).

## Install

```bash
flutter pub add mediapipe_face_mesh
```

## Usage

### Create Face Detector Processor

```dart
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

final faceDetectorProcessor = await FaceDetectorProcessor.create();
```
`model` selects the detector model.

- `FaceDetectionModel.shortRange` (default): for near faces, within
  roughly 2 meters.
- `FaceDetectionModel.fullRange`: dense model for faces farther from the
  camera.
- `FaceDetectionModel.fullRangeSparse`: sparse variant of the full-range
  model.

`maxResults` (default 1) caps the number of detections.

### Create Face Mesh Processor

```dart
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

final faceMeshProcessor = await FaceMeshProcessor.create();
```

`model` selects the mesh model.

- `FaceMeshModel.v2` (default): FaceMesh-V2, upstream
  `face_landmarks_detector.tflite`, the model the current FaceLandmarker
  task uses. Returns 478 landmarks (10 iris points at indices `468..477`).
- `FaceMeshModel.attention`: the official MediaPipe model before V2. Same
  478-landmark layout. In our benchmark its latency is slightly lower than
  V2, and the V2 model card reports improved accuracy over it.
- `FaceMeshModel.v1`: the original mesh, returns 468 landmarks. Pass
  `enableIris: true` to run a separate iris pass and get the 478-landmark
  layout.

### Delegates

Every processor (`FaceDetectorProcessor`, `FaceMeshProcessor`,
`FaceBlendshapesProcessor`) accepts a `delegate` option.

- `FaceMeshDelegate.cpu` (default)
- `FaceMeshDelegate.xnnpack`

On Android they perform about the same. On Windows `cpu` runs the mesh
models 4~5x slower, so `xnnpack` is recommended there.

### Input Formats

Every `process` method takes a `FaceMeshFrame`, which is one of two types.

- `FaceMeshNv21Image`
  Use this for Android camera frames in NV21 layout.
- `FaceMeshImage`
  Use this for RGBA or BGRA buffers such as iOS camera frames, desktop/USB
  (UVC) camera frames, or any decoded image.

Android camera plugins deliver YUV420 in several layouts. `FaceMeshNv21Image`
has helpers that convert them to NV21. See the example camera image adapter.

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
bool _isMeshActive = true; // e.g. driven by a UI toggle

inferenceStreamProcessor
    .process(
      frameController.stream,
      runMeshResolver: (_) => _isMeshActive,
      rotationDegrees: rotationDegrees,
    )
    .listen(_handleInferenceResult, onError: onError);

void _handleInferenceResult(FaceMeshInferenceResult result) {
  _isBusy = false;
  // detectionResult is null on landmark-tracked frames (detector skipped).
  final FaceDetectionResult? detections = result.detectionResult;
  if (detections != null) {
    onDetections(detections);
  }
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

`rotationDegrees` is fixed per subscription. When the camera rotation (or the
input source) changes, re-subscribe with the new value. See the example app
for a complete flow.

#### Landmark tracking

Tracking is on by default. The detector runs only to acquire or re-acquire
a face, and tracked frames report `detectionResult` as null. On face loss
the detector re-acquires on the next frame (`isTracking` reports the
state). Pass `enableLandmarkTracking: false` to run the detector on every
frame, and call `resetTracking()` when switching input sources.

For multi-face behavior, see [Multi-Face Inference](#multi-face-inference).

#### Landmark smoothing

The pipeline smooths output landmarks across frames with a OneEuro filter,
matching the official FaceLandmarker stream-mode behavior. A still face
stops jittering while fast movement passes through with almost no lag.
On by default. Pass `landmarkSmoothing: null` for raw per-frame landmarks.
Tuning options are on `LandmarkSmoothingOptions`.

### Single Inference

Use single-frame inference in one call without a stream processor.

```dart
final pipeline = FaceMeshInferencePipeline(
  detector: faceDetectorProcessor,
  mesh: faceMeshProcessor,
);

final result = pipeline.process(
  nv21Image,
  rotationDegrees: rotationDegrees,
);

final meshResult = result.meshResult;
if (meshResult != null) {
  onResult(meshResult);
}
```

### Geometry and Measurements

`FaceMeshResult` includes helpers for 2D distances and estimated 3D face
geometry.

```dart
// 2D pixel distance between two landmarks
final pixelDistance = meshResult.distancePixels(33, 263);

// 3D geometry estimation (native call; one per frame is typical)
final geometry = meshResult.estimateGeometry();
// Pass actual camera FOV for more accurate centimeter estimates (default: 63°)
// final geometry = meshResult.estimateGeometry(verticalFovDegrees: 72.0);

// Head pose: yaw (left/right), pitch (up/down), roll (tilt)
final pose = geometry.headPose;
// pose.yawDegrees, pose.pitchDegrees, pose.rollDegrees

// Single centimeter distance between two landmarks
final eyeDistanceCm = geometry.distanceCm(33, 263);

// Preset bundle: computes all measurements at once
// faceWidth        234 ↔ 454  cheek-to-cheek
// faceHeight        10 ↔ 152  forehead-to-chin
// eyeOuterDistance  33 ↔ 263  outer eye corners
// eyeInnerDistance 133 ↔ 362  inner eye corners
// interpupillaryDistance 468 ↔ 473  pupils (iris only, else null)
// mouthWidth        61 ↔ 291
// noseWidth         98 ↔ 327 
final measurements = geometry.measurements;
final faceWidthCm = measurements.faceWidth.valueCm;
```

Centimeter values are estimates based on the canonical face geometry model.
Scale accuracy depends on the virtual camera assumption (default vertical FOV
63°) and will vary by device.

To look up landmark indices visually, use https://cornpip.github.io/mediapipe_landmark_viewer/

### Face Blendshapes

Blendshapes are 52 ARKit-style expression coefficients (jaw open, eye blink,
smile, etc.), useful for avatars, AR filters, and expression detection. 
Requires a mesh that returns 478 landmarks.

```dart
final blendshapesProcessor = await FaceBlendshapesProcessor.create();

// Map<FaceBlendshape, double> with values in [0, 1];
// null when the frame had no face.
final blendshapes = blendshapesProcessor.process(meshResult);
if (blendshapes != null) {
  final smile = (blendshapes[FaceBlendshape.mouthSmileLeft]! +
          blendshapes[FaceBlendshape.mouthSmileRight]!) /
      2;
  if (smile > 0.5) {
    // smiling
  }
}
```

### Multi-Face Inference

Multi-face inference tracks each face across frames with a stable `trackId`.
The detector runs only while fewer than `maxMeshFaces` faces are tracked.
The mesh processor must be created with `createForMultiFace(...)`.

```dart
final faceMeshProcessor = await FaceMeshProcessor.createForMultiFace();
final faceDetectorProcessor = await FaceDetectorProcessor.create(
  maxResults: 4,
);
final pipeline = FaceMeshInferencePipeline(
  detector: faceDetectorProcessor,
  mesh: faceMeshProcessor,
);
final inferenceStreamProcessor = FaceMeshInferenceStreamProcessor(pipeline);

inferenceStreamProcessor
    .processMultiFace(
      frameController.stream,
      maxMeshFaces: 2,
      runMeshResolver: (_) => _isMeshActive,
      rotationDegrees: rotationDegrees,
    )
    .listen(_handleMultiInferenceResult, onError: onError);

void _handleMultiInferenceResult(FaceMeshMultiInferenceResult result) {
  // detectionResult is null while all face slots are served by tracking.
  final FaceDetectionResult? detections = result.detectionResult;
  if (detections != null) {
    onDetections(detections);
  }
  for (final TrackedFaceMesh face in result.faces) {
    onFaceMesh(face.trackId, face.mesh); // trackId is stable across frames
  }
}
```

For single-frame multi-face inference, call `pipeline.processMultiFace(...)`
directly.

### Using an External Face Detector

The bundled detector is optional. If you already use another face detector
(e.g. ML Kit), pass its face box to `FaceMeshProcessor.process(...)` as a
`FaceMeshBox` or `NormalizedRect` and skip `FaceDetectorProcessor` entirely.

### Close Resource

Explicitly calling close() when the processors are no longer needed is recommended.

```dart
faceDetectorProcessor.close();
faceMeshProcessor.close();
blendshapesProcessor.close();
```

## Example app

A demo app lives in the `example/` directory at the root of this
repository.

## Notes

- On Flutter older than 3.38.0, a debug `flutter run` on a physical iOS 17+
  device can hang at `Installing and launching...`. Flutter tooling issue,
  see [doc/IOS_DEBUG_RUN.md](doc/IOS_DEBUG_RUN.md).

## License

BSD 3-Clause ([LICENSE](LICENSE)) for this package's own source code. The
bundled TensorFlow Lite runtimes and MediaPipe models are Apache-2.0 and stay
under their own license. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the attributions, the
model sources, and the modifications made to the runtime binaries, and
[LICENSE-APACHE-2.0.txt](LICENSE-APACHE-2.0.txt) for the license text.

Your app does not need to add anything. The package ships a `NOTICES` file, so
`showLicensePage()` lists the bundled components automatically.

This project is not affiliated with or endorsed by Google LLC.
