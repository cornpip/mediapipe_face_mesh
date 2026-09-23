# mediapipe_face_mesh

Face detection and a 478-landmark face mesh pipeline, on device, in a few
milliseconds per frame. Models and the TensorFlow Lite runtime ship inside
the package. The only dependency is `ffi`.

<img src="./readme_img/22.png" alt="face mesh preview" width="300"/> <img src="./readme_img/33.png" alt="multi-face preview" width="300"/>

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
final faceMeshProcessor = await FaceMeshProcessor.create();
```

`model` selects the mesh model.

- `FaceMeshModel.v2` (default): FaceMesh-V2, the current MediaPipe Face
  Landmarker model. Returns 478 landmarks, with 10 iris points at indices
  `468..477`.
- `FaceMeshModel.attention`: the official model before V2. Same
  478-landmark layout.
- `FaceMeshModel.v1`: the original mesh, 468 landmarks. `enableIris: true`
  adds a separate iris pass for the 478-landmark layout.

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

### Camera Frames

Call `process` on the pipeline for every camera frame. Each call runs the
detector and the mesh for that frame and returns the result. The same call
serves a single decoded image.

```dart
final pipeline = FaceMeshInferencePipeline(
  detector: faceDetectorProcessor,
  mesh: faceMeshProcessor,
);

void onCameraFrame(FaceMeshNv21Image frame) {
  final FaceMeshInferenceResult result = pipeline.process(
    frame,
    rotationDegrees: rotationDegrees,
  );
  // detectionResult is null on landmark-tracked frames (detector skipped).
  final FaceDetectionResult? detections = result.detectionResult;
  if (detections != null) {
    onDetections(detections);
  }
  onMeshResult(result.meshResult);
}
```

Pipeline options (`FaceMeshInferencePipeline`), set once.

- `landmarkSmoothing`: output landmarks are smoothed across frames by
  default. Pass `null` for raw per-frame landmarks.
- `enableLandmarkTracking`: on by default. See
  [Mesh Landmark Tracking](#mesh-landmark-tracking).
- `detectionSelector`: picks the face to mesh when the detector finds
  several. Default is the highest score.

Call options (`process`, `processMultiFace`), per frame.

- `rotationDegrees`, `mirrorHorizontal`: transform applied to the frame
  before inference.
- `runMesh: false`: returns detector output without running the mesh.

The full option list and details are in the source (dartdoc).

The call is synchronous and blocks the calling isolate. To keep it off the
UI isolate, see [Background Isolate](#background-isolate).

### Multi-Face Inference

Multi-face inference tracks each face across frames with a stable `trackId`.

```dart
final faceMeshProcessor = await FaceMeshProcessor.create();
final faceDetectorProcessor = await FaceDetectorProcessor.create(
  maxResults: 4, // candidates per detector pass
);
final pipeline = FaceMeshInferencePipeline(
  detector: faceDetectorProcessor,
  mesh: faceMeshProcessor,
);

void onCameraFrame(FaceMeshNv21Image frame) {
  final FaceMeshMultiInferenceResult result = pipeline.processMultiFace(
    frame,
    // Faces tracked at once. The detector runs only while fewer are
    // tracked, to fill the free slots.
    maxMeshFaces: 2,
    rotationDegrees: rotationDegrees,
  );
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

### Background Isolate

`FaceMeshIsolatePipeline` runs the pipeline in a worker isolate. The
factory runs inside the worker, so the processors it creates live there.

```dart
// A top-level or static function, as with Isolate.spawn.
Future<FaceMeshInferencePipeline> createPipeline(FaceMeshModel model) async =>
    FaceMeshInferencePipeline(
      detector: await FaceDetectorProcessor.create(),
      mesh: await FaceMeshProcessor.create(model: model),
    );

final isolatePipeline = await FaceMeshIsolatePipeline.spawn(
  createPipeline,
  FaceMeshModel.v2,
);

Future<void> onCameraFrame(FaceMeshNv21Image frame) async {
  // Unlike the synchronous call, requests queue up in the worker.
  // Drop the frame while it is busy.
  if (isolatePipeline.isBusy) return;
  final FaceMeshInferenceResult result = await isolatePipeline.process(
    frame,
    rotationDegrees: rotationDegrees,
  );
  onMeshResult(result.meshResult);
}
```

Use it like the pipeline. The only difference is that its methods return
`Future`s.

### Close Resources

These are all the objects that need closing.

```dart
pipeline.close(); // closes the detector and mesh it was given
await isolatePipeline.close();
blendshapesProcessor.close();
```

The processors release their native context on garbage collection if
`close()` is skipped, but with no guarantee of when. Close them explicitly.
Unlike the processors, `FaceMeshIsolatePipeline` is never released by
garbage collection, so `close()` is required.

### Mesh Landmark Tracking

Tracking is on by default. The detector runs only to acquire or re-acquire
a face, and tracked frames report `detectionResult` as null. On face loss
the detector re-acquires on the next frame (`isTracking` reports the
state). Tracking also resets when rotation, mirroring, frame size, or the
frame type (NV21 or RGBA) changes. After other source switches it recovers
within a frame or two, or right away with `resetTracking()`. Pass
`enableLandmarkTracking: false` to run the detector on every frame.

For multi-face behavior, see [Multi-Face Inference](#multi-face-inference).

### Geometry and Measurements

`FaceMeshResult` includes helpers for 2D distances and estimated 3D face
geometry.

```dart
final pixelDistance = meshResult.distancePixels(33, 263);

// Native solve, one per frame. Pass the camera's vertical FOV for better
// centimeter estimates (default 63°).
final geometry = meshResult.estimateGeometry();
final pose = geometry.headPose; // yawDegrees, pitchDegrees, rollDegrees
final eyeDistanceCm = geometry.distanceCm(33, 263);
final faceWidthCm = geometry.measurements.faceWidth.valueCm;
```

Centimeter values are estimates and vary by device. The preset
measurements and their landmark indices are listed in
[doc/GEOMETRY_MEASUREMENTS.md](doc/GEOMETRY_MEASUREMENTS.md). To look up
landmark indices visually, use
https://cornpip.github.io/mediapipe_landmark_viewer/

### Face Blendshapes

Blendshapes are 52 ARKit-style expression coefficients (jaw open, eye blink,
smile, etc.), useful for avatars, AR filters, and expression detection.
Requires a mesh that returns 478 landmarks (`FaceMeshResult.hasIris`).

```dart
final blendshapesProcessor = await FaceBlendshapesProcessor.create();

// FaceBlendshapes with values in [0, 1]. Null when the frame had no face.
final blendshapes = blendshapesProcessor.process(meshResult);
if (blendshapes != null) {
  final left = blendshapes[FaceBlendshape.mouthSmileLeft];
  final right = blendshapes[FaceBlendshape.mouthSmileRight];
  if ((left + right) / 2 > 0.5) {
    // smiling
  }
}
```

### Using an External Face Detector

The bundled detector is optional. If you already use another face detector
(e.g. ML Kit), pass its face box to `FaceMeshProcessor.process(...)` and skip
`FaceDetectorProcessor` entirely.

```dart
final box = FaceMeshBox.fromLTWH(
  left: face.left, top: face.top, width: face.width, height: face.height,
);
final meshResult = faceMeshProcessor.process(
  image,
  box: box,
  // The image is the raw camera buffer, so pass the rotation the detector
  // applied to it. The box is already in that rotated frame.
  rotationDegrees: rotationDegrees,
);
```

### Overlay Painters

Two `CustomPainter`s draw results on a preview. Each takes the same
`rotationDegrees` and `mirrorHorizontal` used for the preview. Both have
`fromInference(FaceMeshInferenceResult)` and
`fromMultiInference(FaceMeshMultiInferenceResult)` constructors that take
the pipeline result as is.

- `FaceMeshPainter` (`face_mesh_painter.dart`): landmarks, mesh edges, iris.
- `FaceDetectionPainter` (`face_detection_painter.dart`): detector boxes
  and the ROI the mesh ran on, with an optional label per face.

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
