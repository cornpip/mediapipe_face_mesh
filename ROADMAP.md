# Roadmap

## Landmark smoothing

Add an optional OneEuro-style temporal filter on the landmark coordinates,
matching the official MediaPipe video-mode behavior. The existing
`enableSmoothing` only stabilizes the tracked ROI (the crop fed to the model);
per-point output noise still reaches consumers such as blendshapes and head
pose estimation.

Scope:

- `enableLandmarkSmoothing` option on `FaceMeshProcessor.create` (start
  opt-in; consider making it the default after field validation)
- OneEuro filter state per landmark (478 × x/y/z), reset on tracking loss,
  re-acquisition, and orientation changes
- Frame timing: estimate dt from an internal clock for real-time streams, or
  accept an optional timestamp on `process()`/`processNv21()`
- Scale-aware filtering: normalize filter strength by face size so distant
  faces are not over-filtered (matches the official graph)
- Tune `min_cutoff`/`beta` on-device for the stillness-vs-expression-latency
  trade-off

## Align tracking-confidence semantics with the official graph

When the tracked-frame confidence drops below `minTrackingConfidence`, the
official graph drops the ROI and re-detects; this package currently freezes
the last ROI and keeps returning landmarks, so a `minTrackingConfidence`
raised above the face-presence threshold (0.5) creates a window where the ROI
stops following the face. Harmless at the default configuration (the window
is empty), documented on `FaceMeshProcessor.create`.

Scope:

- Expose the native tracking state through FFI (e.g.
  `mp_face_mesh_is_tracking`) so `FaceMeshInferencePipeline` no longer infers
  it from empty landmarks — the prerequisite for dropping the ROI safely
  (native header change, ffigen regeneration)
- On tracking-confidence failure, invalidate the ROI and have the pipeline
  re-acquire via the detector on the same or next frame instead of meshing a
  full-frame `DefaultRect`
- Expose `minFacePresenceConfidence` from Dart (already present in the native
  options struct but never set)

## Native multi-face API (single frame upload)

The multi-face flow runs one `process()` call per face, and each call copies
the full frame into native memory and frees it again — N tracked faces cost
N full-frame copies per frame (plus one for the detector frame). Not a leak,
but avoidable allocation and memcpy churn that grows with face count.

Scope:

- Native entry point that accepts the frame once plus a list of ROIs and
  returns one result per ROI (e.g. `mp_face_mesh_process_rois`), for both
  RGBA and NV21 inputs (native header change, ffigen regeneration)
- Route `FaceMeshInferencePipeline`'s multi-face tracked/acquisition mesh
  calls and `FaceMeshProcessor.processMultiFace`/`processNv21MultiFace`
  through it
- Benchmark per-frame allocation and latency against the per-face path with
  2–4 tracked faces before switching

## Attention-based landmark refinement model

Eye/iris refinement currently follows the legacy official pipeline: base
468-landmark mesh plus a separate `iris_landmark.tflite` pass merged into the
result. The current official FaceLandmarker uses the unified
`face_landmark_with_attention` model, which refines lips, eyes, and irises in
one inference with better accuracy around those regions.

Scope:

- Bundle and support the attention model as an alternative to the
  mesh + iris two-pass setup (model availability/licensing check first)
- Map its outputs to the existing 478-landmark index layout so `enableIris`
  consumers and blendshapes keep working unchanged
- Compare accuracy/latency against the current two-pass pipeline on-device
  before switching any default
