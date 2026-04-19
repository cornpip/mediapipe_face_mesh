# mediapipe_face_mesh

MediaPipe Face Mesh for Flutter — 468 3D landmarks with mesh triangulation.  
Exposes a simple API for running single snapshots or continuous camera streams.  

Bundles the [MediaPipe face mesh TFLite model](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md) and prebuilt TensorFlow Lite C runtime binaries for Android (`arm64-v8a`, `x86_64`) and iOS — no extra setup required.

- Optional XNNPACK or GPU (V2) delegates for faster inference.
- Supports RGBA/BGRA buffers and Android NV21 camera frames.
- ROI helpers (`FaceMeshBox`, `NormalizedRect`) to limit processing to face regions.
- Stream processor utilities to consume frames sequentially and deliver `FaceMeshResult` updates.

Note: Face detection is not included.  
If you need dynamic ROIs, use a face detector (e.g. [google_mlkit_face_detection](https://pub.dev/packages/google_mlkit_face_detection)) before calling this package.

## Install

```bash
flutter pub add mediapipe_face_mesh
```

## Usage

Two ways to use it:
1. Provide one frame at a time (Single Frame Inference)
2. Provide a stream of frames (Frame Stream Inference)

Both approaches run the same per-frame computation. The only difference is who drives the frame flow: you push each frame manually, or you hand off a stream and receive results as they are emitted.

### Create
```dart
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

final faceMeshProcessor = await FaceMeshProcessor.create(
  delegate: FaceMeshDelegate.xnnpack, // FaceMeshDelegate.cpu is default
);
```

### Single Frame Inference
```dart
// Android — NV21
if (Platform.isAndroid) {
  final result = faceMeshProcessor.processNv21(
    nv21Image,
    box: FaceMeshBox.fromLTWH(left: ..., top: ..., width: ..., height: ...),
    boxScale: 1.2,
    boxMakeSquare: true,
    rotationDegrees: rotationCompensation,
  );
}

// iOS — BGRA
if (Platform.isIOS) {
  final result = faceMeshProcessor.process(
    bgraImage,
    box: FaceMeshBox.fromLTWH(left: ..., top: ..., width: ..., height: ...),
    boxScale: 1.2,
    boxMakeSquare: true,
    rotationDegrees: rotationCompensation,
  );
}
```

### Frame Stream Inference
```dart
final streamProcessor = FaceMeshStreamProcessor(faceMeshProcessor);

// Android — NV21 stream
final nv21Controller = StreamController<FaceMeshNv21Image>();
streamProcessor
    .processNv21(
      nv21Controller.stream,
      boxResolver: (frame) => resolveBox(frame),
      boxScale: 1.2,
      boxMakeSquare: true,
      rotationDegrees: rotationDegrees,
    )
    .listen(onResult, onError: onError);

// iOS — BGRA stream
final bgraController = StreamController<FaceMeshImage>();
streamProcessor
    .process(
      bgraController.stream,
      boxResolver: (frame) => resolveBox(frame),
      boxScale: 1.2,
      boxMakeSquare: true,
      rotationDegrees: rotationDegrees,
    )
    .listen(onResult, onError: onError);
```

## Example

The __[example included in this package](https://github.com/cornpip/mediapipe_face_mesh/tree/master/example)__ streams live camera frames, detects face bounding boxes using `google_mlkit_face_detection`, and passes them to `mediapipe_face_mesh` for landmark inference. The resulting 468 landmarks are rendered as a triangulated mesh polygon overlay on the camera preview.

<img src="./readme_img/1.png" alt="app_image_1" width="300"/>
<img src="./readme_img/2.png" alt="app_image_2" width="300"/>

## Detail

### FaceMeshProcessor.create parameter

```dart
final faceMeshProcessor = await FaceMeshProcessor.create(
  delegate: FaceMeshDelegate.xnnpack, // FaceMeshDelegate.cpu is default
);
```

- `threads`: number of CPU threads used by TensorFlow Lite. Increase it to speed
  up inference on multi-core devices, keeping thermal/power trade-offs in mind. (default 2)
- `delegate`: choose between CPU, XNNPACK, or GPU (V2) delegates. Default is `FaceMeshDelegate.cpu`.
- `minDetectionConfidence`: threshold for the initial face detector. Lowering it
  reduces missed detections but may increase false positives (default 0.5).
- `minTrackingConfidence`: threshold for keeping an existing face track alive.
  Higher values make tracking stricter but can drop faces sooner (default 0.5).
- `enableSmoothing`: toggles MediaPipe's temporal smoothing between frames.
  Keeping it `true` (default) reduces jitter but adds inertia; set `false` for
  per-frame responsiveness when you don't reuse tracking context.
- `enableRoiTracking`: enables internal ROI tracking between frames. When set
  to `false`, calls that omit `roi`/`box` always run full-frame inference.

Always remember to call `close()` on the processor when you are done.

### FaceMeshProcessor.process parameter

```dart
result = faceMeshProcessor.process(
  image,
  box: box,
  boxScale: 1.2,
  boxMakeSquare: true,
  rotationDegrees: rotationCompensation,
);
```
- `image`: `FaceMeshImage` containing an RGBA/BGRA pixel buffer. The processor
  copies the data into native memory, so the underlying bytes can be reused
  immediately after the call returns.
- `roi`: optional `NormalizedRect` that describes the region of interest
  in normalized 0..1 coordinates (MediaPipe layout: `xCenter`, `yCenter`,
  `width`, `height`, `rotation`). Use this when you precompute ROIs yourself;
  no extra clamping, scaling, or squaring is performed inside the plugin. Cannot be combined with `box`.
- `box`: optional `FaceMeshBox` in pixel space. When provided, it is converted
  internally into a normalized rect, clamped to the image bounds, optionally
  squarified, and then scaled by `boxScale`. Helps limit work to the detected
  face instead of the entire frame.
- `boxScale`: multiplicative expansion/shrink factor applied to the ROI derived
  from `box`. Values >1.0 pad the box (default 1.2). Must be positive.
- `boxMakeSquare`: when `true`, the converted ROI uses the max-side length for
  both width and height so the downstream Face Mesh graph gets a square crop.
  Set `false` to retain the original aspect ratio of the box.
- `rotationDegrees`: informs the native graph about the orientation of the
  provided pixels. Only 0/90/180/270 are allowed; logical width/height swap
  automatically, so ROIs remain aligned with upright faces.
- `mirrorHorizontal`: mirrors the input crop horizontally before inference so
  the returned landmarks already align with mirrored front-camera previews.

If both `roi` and `box` are omitted, the processor uses its internal ROI
tracking state (or the full frame if `enableRoiTracking` is disabled). Passing
both results in an `ArgumentError`.

The same parameter rules apply to `processNv21`, using the NV21 image wrapper
instead of an RGBA/BGRA buffer.

### FaceMeshStreamProcessor.process parameter

```dart
streamProcessor
  .process(
    bgraController.stream,
    boxResolver: resolveBox,
    boxScale: 1.2,
    boxMakeSquare: true,
    rotationDegrees: rotationDegrees,
  )
  .listen(onResult, onError: onError);
```

- `frames`: `Stream<FaceMeshImage>` source. Each frame is awaited sequentially
  before being passed to `faceMeshProcessor.process`.
- `roi`: matches the `faceMeshProcessor.process` semantics (a precomputed
  normalized rectangle) and cannot be combined with `boxResolver`.
- `boxResolver`: optional callback that returns a `FaceMeshBox` per frame,
  which is then processed through the same clamp/scale/square pipeline used by `faceMeshProcessor.process`.

`FaceMeshStreamProcessor.process()` internally invokes `faceMeshProcessor.process`
for every frame, so the ROI/box/rotation/mirroring options behave identically. The
only difference is that it consumes an incoming `Stream<FaceMeshImage>` and forwards
each awaited frame with the parameters you provide (or the per-frame `boxResolver`).

`.processNv21` follows the same flow, but operates on `Stream<FaceMeshNv21Image>` sources
and forwards them to `faceMeshProcessor.processNv21`.

### Output (FaceMeshResult)

```dart
class FaceMeshResult {
  /// All face landmarks returned by the native graph.
  final List<FaceMeshLandmark> landmarks;

  /// Triangles describing the mesh topology.
  final List<MpFaceMeshTriangle> triangles;

  /// Normalized rectangle covering the detected face.
  final NormalizedRect rect;

  /// Confidence score reported by MediaPipe.
  final double score;

  /// Width of the image used during inference.
  final int imageWidth;

  /// Height of the image used during inference.
  final int imageHeight;
}

class FaceMeshLandmark {
  /// Horizontal coordinate normalized to [0, 1].
  final double x;

  /// Vertical coordinate normalized to [0, 1].
  final double y;

  /// Depth relative to the camera in canonical MediaPipe units.
  final double z;
}

class MpFaceMeshTriangle {
  /// Indices into the full landmark list (length 3).
  final List<int> indices;

  /// Landmark points referenced by [indices] (length 3).
  final List<FaceMeshLandmark> points;
}
```

- `landmarks`: list of `FaceMeshLandmark` points (468 for the base
  Face Mesh model). `x`/`y` are normalized relative to the input frame size;
  values may extend slightly beyond 0..1 because the ROI can be
  expanded/rotated and the native code clamps to a wider range (-0.5..1.5).
  `z` is the MediaPipe depth (negative values are closer to the camera in
  MediaPipe's convention).
- `triangles`: list of `MpFaceMeshTriangle` entries derived from the official
  MediaPipe face mesh tesselation topology. Each triangle references three
  landmark indices and their corresponding points.
- `rect`: `NormalizedRect` describing the detected face ROI in normalized space
  relative to the input frame size (`xCenter`, `yCenter`, `width`, `height`,
  `rotation` in radians, clockwise). `width`/`height` can exceed 1.0 when the
  ROI is expanded or rotated.
- `score`: confidence score reported by the native graph (0..1). If the model
  does not provide a score output tensor, the plugin returns `1.0`.
- `imageWidth`/`imageHeight`: input frame size used for inference (after applying
  `rotationDegrees`, so 90/270 swap width/height).
