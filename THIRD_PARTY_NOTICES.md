# Third-Party Notices

`mediapipe_face_mesh` is licensed under the BSD 3-Clause License (see
[`LICENSE`](LICENSE)). That license covers this project's own source code.

The package also redistributes third-party components that remain under their
own licenses. Those components and their notices are listed below. A verbatim
copy of the Apache License, Version 2.0 is included as
[`LICENSE-APACHE-2.0.txt`](LICENSE-APACHE-2.0.txt).

Apps that depend on this package pick these notices up automatically. The
`NOTICES` file in the package root carries the same attribution in the format
Flutter's license collector reads, so `showLicensePage()` in a consuming app
lists the bundled Apache-2.0 components alongside this package's BSD-3 license.

## TensorFlow Lite

Copyright The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
Upstream: https://github.com/tensorflow/tensorflow

Redistributed in this package:

| Path | Contents |
| --- | --- |
| `src/include/tensorflow/**` | C/C++ headers from TensorFlow 2.19.0, a headers-only subset of the upstream tree, used to compile this package's native sources |
| `android/src/main/jniLibs/arm64-v8a/libtensorflowlite_c.so` | Prebuilt TensorFlow Lite C runtime |
| `android/src/main/jniLibs/x86_64/libtensorflowlite_c.so` | Prebuilt TensorFlow Lite C runtime |
| `ios/Frameworks/TensorFlowLiteC.xcframework` | Prebuilt TensorFlow Lite C runtime (`ios-arm64`, `ios-arm64_x86_64-simulator`) |
| `windows/blobs/tensorflowlite_c.dll` | Prebuilt TensorFlow Lite C runtime (x64) |

Each redistributed header retains its original Apache-2.0 file notice.

### Modifications (Apache-2.0 section 4(b))

The prebuilt runtime binaries listed above are **not** stock upstream releases.
They were rebuilt from source so that the TensorFlow Lite C API's default
operator resolver also registers the MediaPipe custom operators required by
`face_landmark_with_attention.tflite`:

`TransformTensorBilinear`, `TransformLandmarks`, `Landmarks2TransformMatrix`,
`MaxPoolingWithArgmax2D`, `MaxUnpooling2D`, `Convolution2DTransposeBias`,
`Resampler`.

- **Android and iOS**: built from the MediaPipe Bazel workspace, which pins its
  own `org_tensorflow` commit and applies MediaPipe's TensorFlow patches. A
  local `tflite::CreateOpResolver()` returns `BuiltinOpResolver` combined with
  MediaPipe's `MediaPipe_RegisterTfLiteOpResolver`.
- **Windows**: built from the TensorFlow 2.19.0 CMake tree with MediaPipe's CPU
  kernels for those operators added to the build and registered in
  `tensorflow/lite/core/create_op_resolver_with_builtin_ops.cc`. Local build
  fixes were also applied for source encoding, shared-library export
  configuration, and XNNPACK delegate symbol exports.

No changes were made to TensorFlow Lite's inference behavior beyond operator
registration and the build configuration described above. The vendored headers
are unmodified upstream files.

### Components linked into the prebuilt runtime binaries

The binaries above statically link TensorFlow Lite's own third-party
dependencies. These include, but are not limited to, XNNPACK (BSD 3-Clause),
FlatBuffers (Apache-2.0), Abseil (Apache-2.0), Eigen (MPL-2.0 with
BSD-licensed parts), ruy (Apache-2.0), gemmlowp (Apache-2.0), pthreadpool
(BSD 2-Clause), cpuinfo (BSD 2-Clause), FP16 (MIT), farmhash (MIT), and fft2d
(Ooura, freely distributable). For the authoritative set, see the dependency
licenses in the TensorFlow 2.19.0 source tree:
https://github.com/tensorflow/tensorflow/tree/v2.19.0/third_party

## MediaPipe

Copyright The MediaPipe Authors.
Licensed under the Apache License, Version 2.0.
Upstream: https://github.com/google-ai-edge/mediapipe

### Bundled models

The following files in `assets/models/` are MediaPipe models. Their contents are
unmodified; the packaging notes below describe how each one is bundled.

| File | Upstream model | Packaging |
| --- | --- | --- |
| `face_detection_short_range.tflite` | BlazeFace short-range face detection | as published |
| `face_detection_full_range.tflite` | BlazeFace full-range face detection | as published |
| `face_detection_full_range_sparse.tflite` | BlazeFace full-range sparse face detection | as published |
| `mediapipe_face_mesh.tflite` | Face landmark model | renamed |
| `face_landmark_with_attention.tflite` | Attention-mesh face landmark model | as published |
| `iris_landmark.tflite` | Iris landmark model | as published |
| `face_blendshapes.tflite` | Blendshapes model | extracted from the Face Landmarker task bundle |

Sources:

- Model index:
  https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md
- Face Landmarker task bundle:
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

### Derived data

`src/mediapipe_face_geometry_data.h` is generated from
`mediapipe/modules/face_geometry/data` (canonical face model vertices and
Procrustes solver weights) and reformatted as C++ constants. The numeric data
is MediaPipe's; only the file format differs.

### Custom operator kernels

The CPU kernels for the seven MediaPipe custom TensorFlow Lite operators listed
earlier originate from `mediapipe/util/tflite/operations` and are compiled into
the bundled runtime binaries.

### Scope

The verbatim MediaPipe material redistributed here is the model files, the
geometry data, and the custom operator kernels described above. The pipeline
code in `src/*.cc` is written for this package against MediaPipe's documented
graph and calculator behavior, with small constant tables mirroring MediaPipe's
landmark index sets so the outputs match.

## Trademarks

"MediaPipe" and "TensorFlow" are trademarks of Google LLC. This project is not
affiliated with, endorsed by, or sponsored by Google LLC. Those names are used
descriptively to identify the upstream models and runtime that this package
bundles. Nothing in the BSD 3-Clause License or the Apache License, Version 2.0
grants trademark rights.

## Repository-only assets

The following are present in the source repository but excluded from the
published package by `.pubignore`:

- `bench/assets/portrait.jpg`: MediaPipe test asset, Apache-2.0.
- `bench/assets/bench_face_10s.mp4`: derived from NASA imagery, public domain
  as a US government work.

See `bench/assets/SOURCES.md` for details.
