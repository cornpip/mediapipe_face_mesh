# Roadmap

## Changes held for the next major version (3.0.0)

Breaking or behavior-changing items batched for the next major, called out
in migration notes:

- Landmark smoothing default-on (`landmarkSmoothing: null` as the opt-out).
  The official FaceLandmarker smooths by default in stream mode; 2.4.0
  shipped it opt-in. Caveats and tuning notes live in
  `mediapipe_docs/landmark-smoothing-notes.md`.
- Default mesh model becomes `FaceMeshModel.v2`.
- Mesh model selection unified on the `model` parameter (`FaceMeshModel`,
  shipped in 2.8.0): deprecate `enableAttentionMesh` and make `model` the
  only way to pick the mesh model.
- Remove `FaceMeshDelegate.gpuV2` (deprecated in 2.6.0; benchmarks showed
  the GPU delegate several times slower than CPU/XNNPACK for these models).
