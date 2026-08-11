# Roadmap

## Changes held for the next major version

Breaking or behavior-changing items batched for the next major, called out
in migration notes:

- Landmark smoothing default-on (`landmarkSmoothing: null` as the opt-out).
  The official FaceLandmarker smooths by default in stream mode; 2.4.0
  shipped it opt-in. Caveats and tuning notes live in
  `mediapipe_docs/landmark-smoothing-notes.md`.
- Attention mesh default-on (`enableAttentionMesh` currently defaults to
  false; README already recommends enabling it).
- Remove `FaceMeshDelegate.gpuV2` (deprecated in 2.6.0; benchmarks showed
  the GPU delegate several times slower than CPU/XNNPACK for these models).

## FaceMesh-V2 model (opt-in)

The upstream FaceLandmarker task bundle has moved to FaceMesh-V2: 256x256
input, 478 landmarks with irises built in, fp16 weights, and no custom ops.
That means XNNPACK can delegate the whole graph (unlike the attention
model) and fp16 runs natively on ARMv8.2+ CPUs. Evaluate accuracy and speed
against the current attention path, then add it as an opt-in model choice.

## Derived metrics utilities

Pure-Dart helpers on top of the existing landmark/iris/geometry outputs, so
apps get answers instead of raw points: eye aspect ratio and blink detection,
approximate gaze direction from the iris landmarks, head pose as
yaw/pitch/roll, and simple mouth-open/smile scalars. No native changes.
