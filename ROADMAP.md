# Roadmap

## Deprecated in a 3.x release, removed in 4.0.0

- `FaceMeshInferenceStreamProcessor`. Deprecated in 3.1.0. It was a thin
  wrapper around `FaceMeshInferencePipeline.process`, so call that per
  frame instead.

## Planned

- Moving the native build to build hooks (`package_ffi`). Revisit when one
  of these holds.
  - `native_toolchain_c` reaches 1.0.0.
  - Flutter deprecates building `ffiPlugin` packages.
