# Roadmap

## Landmark smoothing: default-on at 3.0.0

The official FaceLandmarker enables landmark smoothing by default in stream
mode; 2.4.0 shipped it opt-in so a minor update does not change existing
users' output. After field validation, flip the default at the next major
version with `landmarkSmoothing: null` as the opt-out, and call the behavior
change out in the migration notes. Caveats and tuning notes live in
`mediapipe_docs/landmark-smoothing-notes.md`.

## Derived metrics utilities

Pure-Dart helpers on top of the existing landmark/iris/geometry outputs, so
apps get answers instead of raw points: eye aspect ratio and blink detection,
approximate gaze direction from the iris landmarks, head pose as
yaw/pitch/roll, and simple mouth-open/smile scalars. No native changes.

## Windows: attention mesh support

Ship a Windows TensorFlow Lite runtime that includes the MediaPipe custom ops
so `enableAttentionMesh` works there too (since 2.3.0 it throws
`UnsupportedError` on Windows). Key constraint: the runtime must be built from
the MediaPipe workspace — like the bundled Android/iOS runtimes — because the
custom ops do not exist in stock TensorFlow.

## GPU delegate

Make `FaceMeshDelegate.gpuV2` actually engage: the GPU delegate is a separate
binary that this package has never bundled, so today the request silently
falls back to CPU. Key constraint: build the delegate from the MediaPipe
workspace so the attention model's custom ops can run on GPU as well. Windows
stays CPU/XNNPACK.

## Note: attention path and XNNPACK

Not planned work, just a known behavior: XNNPACK cannot run the attention
model's custom ops, so TFLite partitions the graph and those nodes run on the
reference CPU kernels — the delegate accelerates less of the attention path
than it does the base mesh.
