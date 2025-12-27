# mediapipe_face_mesh

Flutter/FFI bindings around the MediaPipe Face Mesh CPU graph.  
The package bundles the default MediaPipe `.tflite` model and exposes a simple
API for running single snapshots or continuous camera streams.

## Highlights

- CPU inference with the TensorFlow Lite C runtime (no GPU dependency).
- Works with RGBA/BGRA buffers and Android NV21 camera frames.
- ROI helpers (`FaceMeshBox`, `NormalizedRect`) to limit processing to a face.
- Stream processor utilities to consume frames sequentially and deliver
  `FaceMeshResult` updates.

## Installation

```bash
flutter pub add mediapipe_face_mesh
```

The plugin bundles the native binaries and a default model, so no extra setup is
required. When creating the processor you can adjust:

- `threads`, `minDetectionConfidence`, `minTrackingConfidence`,
  `enableSmoothing` to tune performance/accuracy.

The packaged MediaPipe model (`assets/models/mediapipe_face_mesh.tflite`) is
always used to keep deployments consistent.

Parameter hints:

- `minDetectionConfidence`: threshold for the initial face detector. Lowering it
  reduces missed detections but may increase false positives (default 0.5).
- `minTrackingConfidence`: threshold for keeping an existing face track alive.
  Higher values make tracking stricter but can drop faces sooner (default 0.5).
- `threads`: number of CPU threads used by TensorFlow Lite. Increase it to speed
  up inference on multi-core devices, keeping thermal/power trade-offs in mind.
  
Always remember to call `close()` on the processor when you are done.

## Usage

### Single image processing

```dart
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

final FaceMeshProcessor processor = await FaceMeshProcessor.create(
  threads: 4,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.6,
);

// Convert your source image into an RGBA/BGRA byte buffer.
final FaceMeshImage frame = FaceMeshImage(
  pixels: rgbaBytes, // Uint8List, length >= width * height * 4
  width: imageWidth,
  height: imageHeight,
  pixelFormat: FaceMeshPixelFormat.rgba,
);

final FaceMeshResult result = processor.process(
  frame,
  // Optionally restrict processing to a face-sized ROI.
  box: FaceMeshBox.fromLTWH(
    left: detectedBox.left,
    top: detectedBox.top,
    width: detectedBox.width,
    height: detectedBox.height,
  ),
  rotationDegrees: 0,
  mirrorHorizontal: false,
);

// Convert landmarks into pixel coordinates.
final Iterable<Offset> offsets = result.landmarks.asMap().entries.map(
  (entry) => faceMeshLandmarkOffset(
    result,
    entry.value,
    targetSize: Size(imageWidth.toDouble(), imageHeight.toDouble()),
  ),
);

processor.close();
```

- Pass `FaceMeshImage` for RGBA/BGRA inputs or `FaceMeshNv21Image` for NV21.
- Provide either `roi` or `box` to limit the search area (or neither for
  full-frame processing).
- Use `faceMeshBoundingRect`, `faceMeshLandmarkOffset`, or
  `faceMeshLandmarksOffsets` to project normalized coordinates into pixels.

### Streaming camera frames

`FaceMeshStreamProcessor` wraps `FaceMeshProcessor` so that each frame is
processed sequentially. This is useful when working with a continuous camera
stream (e.g. Android's `CameraController` that yields NV21 buffers).

```dart
final FaceMeshProcessor processor = await FaceMeshProcessor.create();
final FaceMeshStreamProcessor streamProcessor =
    createFaceMeshStreamProcessor(processor);

// `cameraFrames` is a Stream<FaceMeshNv21Image> created from your camera plugin.
streamProcessor
    .processNv21(
      cameraFrames,
      // Optionally provide a dynamic box per frame.
      boxResolver: (FaceMeshNv21Image frame) => previousBox,
      rotationDegrees: 90,
      mirrorHorizontal: true,
    )
    .listen((FaceMeshResult result) {
      final Rect faceRect = faceMeshBoundingRect(
        result,
        targetSize: const Size(720, 1280),
      );
      // Update UI or tracking state here.
    });
```

Use `processImages` when your stream already emits `FaceMeshImage` buffers.
The helper validates that exactly one of `roi` or `boxResolver` is provided
and forwards the rest of the parameters to the underlying processor.

## Example

A minimal Flutter example that runs inference on an asset lives under
`example/`. Run it with:

```bash
cd example
flutter run
```

The sample demonstrates loading an asset into `FaceMeshImage`, running a single
inference, and drawing the resulting landmarks.
