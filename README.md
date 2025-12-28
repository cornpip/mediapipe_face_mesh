# mediapipe_face_mesh

Flutter/FFI bindings around the MediaPipe Face Mesh graph with optional XNNPACK/GPU delegates.  
The plugin bundles the native binaries and a default model, so no extra setup is required.  
exposes a simple API for running single snapshots or continuous camera streams.

- TensorFlow Lite C runtime loaded dynamically, with optional XNNPACK or GPU (V2) delegates.
- Works with RGBA/BGRA buffers and Android NV21 camera frames.
- ROI helpers (`FaceMeshBox`, `NormalizedRect`) to limit processing to a face.
- Stream processor utilities to consume frames sequentially and deliver
  `FaceMeshResult` updates.

## Usage

```bash
flutter pub add mediapipe_face_mesh
```

### create
```dart
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

final FaceMeshProcessor processor = await FaceMeshProcessor.create();
```

### single image porcessing
```dart
          if (Platform.isAndroid) {
            meshResult = _runFaceMeshOnAndroidNv21(
              mesh: _faceMeshProcessor,
              cameraImage: cameraImage,
              face: faces.first,
              rotationCompensationDegrees: rotationCompensation,
            );
          } else if (Platform.isIOS) {
            meshResult = _runFaceMeshOnIosBgra(
              mesh: _faceMeshProcessor,
              cameraImage: cameraImage,
              face: faces.first,
              rotationCompensationDegrees: rotationCompensation,
            );
          }
```

### streaming camera frames
```dart
    if (Platform.isAndroid) {
      _nv21StreamController = StreamController<FaceMeshNv21Image>();
      _meshStreamSubscription = _faceMeshStreamProcessor
          .processNv21(
            _nv21StreamController!.stream,
            boxResolver: _resolveFaceMeshBoxForNv21,
            boxScale: 1.2,
            boxMakeSquare: true,
            rotationDegrees: rotationDegrees,
          )
          .listen(_handleMeshResult, onError: _handleMeshError);
    } else if (Platform.isIOS) {
      _bgraStreamController = StreamController<FaceMeshImage>();
      _meshStreamSubscription = _faceMeshStreamProcessor
          .processImages(
            _bgraStreamController!.stream,
            boxResolver: _resolveFaceMeshBoxForBgra,
            boxScale: 1.2,
            boxMakeSquare: true,
            rotationDegrees: rotationDegrees,
          )
          .listen(_handleMeshResult, onError: _handleMeshError);
    }
```

## Example

The example demonstrates loading an asset into `FaceMeshImage`, running a single inference, and drawing the resulting landmarks.   
If you need a camera-based example, check https://github.com/cornpip/flutter_vision_ai_demos.git which streams camera frames instead of using an asset.

## Reference

### Model asset

The plugin ships with `assets/models/mediapipe_face_mesh.tflite`, taken from the Face Landmark model listed in Google’s official collection: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md.

### TensorFlow Lite
The `src/include/tensorflow/lite` and `src/include/tensorflow/compiler` directories are copied directly from the official TensorFlow repository: https://github.com/tensorflow/tensorflow/tree/master/tensorflow.

## detail

### .create parameter

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

Always remember to call `close()` on the processor when you are done.
