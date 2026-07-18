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
  re-acquisition, and orientation changes — `FaceMeshProcessor.isTracking`
  (added in 2.2.0) is the reset signal
- Frame timing: estimate dt from an internal clock for real-time streams, or
  accept an optional timestamp on `process()`/`processNv21()`
- Scale-aware filtering: normalize filter strength by face size so distant
  faces are not over-filtered (matches the official graph)
- Tune `min_cutoff`/`beta` on-device for the stillness-vs-expression-latency
  trade-off

## Attention mesh: partial XNNPACK acceleration

The attention model (`face_landmark_with_attention`) shipped in 2.1.0 as the
opt-in `enableAttentionMesh`. XNNPACK cannot run its three MediaPipe custom ops,
so TFLite partitions the graph and those nodes fall back to the reference CPU
kernels — `FaceMeshDelegate.xnnpack` accelerates only the rest of the model on
the attention path. It works, but the delegate buys less here than it does for
the base mesh. Worth knowing before chasing attention-path latency; GPU is a
separate matter, see "GPU delegate" below.

## GPU delegate

`FaceMeshDelegate.gpuV2` is effectively a no-op right now. The bundled
`libtensorflowlite_c.so` / `TensorFlowLiteC.framework` do not export
`TfLiteGpuDelegateV2Create` — in TFLite the GPU delegate is a *separate*
binary (`libtensorflowlite_gpu_delegate.so` on Android, the Metal delegate
framework on iOS) and this package has never bundled one. `tflite_runtime.h`
loads the symbol with `LoadSymbolOptional`, does not find it, and the request
silently degrades to CPU (or fails outright when `allowDelegateFallback` is
false). This predates the attention work and affects every model, not just the
attention one.

Scope:

- Build and bundle the GPU delegate binaries (Android per ABI, iOS Metal) and
  wire them into `tflite_runtime.h` so `gpuV2` actually engages
- Build them **from the MediaPipe workspace**, not from stock TensorFlow:
  MediaPipe's `org_tensorflow_custom_ops.diff` adds GPU support for the
  attention model's custom ops (`custom_parsers.cc`, the GL kernels under
  `gpu/gl/kernels/mediapipe/`, and the compute tasks). A stock GPU delegate
  does not know `TransformTensorBilinear` / `TransformLandmarks` /
  `Landmarks2TransformMatrix`, so it could not run the attention model on GPU.
  The same custom-op resolver work already done for the C API applies here
- Until the delegate ships, make the degradation observable rather than silent
  (surface it through the active-delegate getter, or document it plainly)
- Benchmark GPU against CPU/XNNPACK for both the base mesh and the attention
  model, including how graph partitioning behaves when the delegate cannot take
  every node
