# Roadmap

## Landmark smoothing

Add an optional OneEuro-style temporal filter on the landmark coordinates,
matching the official MediaPipe video-mode behavior. The existing
`enableSmoothing` only stabilizes the tracked ROI; per-point output noise
still reaches consumers such as blendshapes and head pose estimation.

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
