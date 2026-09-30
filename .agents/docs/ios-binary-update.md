# Apple bundled binary update

Apple builds compile against the headers bundled in the xcframework, not
against `src/include`, so the two must describe the same runtime.

When a bundled iOS or macOS binary is replaced:

- Re-sync `Headers/TensorFlowLiteC/` from `src/include` in every slice.
  Stale headers silently disagree with the binary about struct layout.
- The bundled headers are flattened copies: their includes name the file
  only (`#include "common.h"`), and the GPU delegate headers are renamed
  `gpu_delegate.h` and `gpu_delegate_options.h`. Keep those edits.
- Keep `Headers/` identical in every slice (`diff -r`). `TensorFlowLiteC.h`
  is this package's umbrella, not an upstream file; add new headers to it.
- Ship the runtime as a library xcframework (`libTensorFlowLiteC.a`, a
  static archive per slice, built with `libtool -static`), not a framework:
  Xcode tries to embed a framework from a Swift package binary target and
  fails on a static one.
