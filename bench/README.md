# Benchmark suite

Minimal apps, one per measurement target. Separate apps because each
package bundles its own native binaries and because app-size deltas need
isolated builds. All bundle identical assets so size deltas cancel out.

| dir | app | version | purpose |
| --- | --- | --- | --- |
| `mine/` | mediapipe_face_mesh | path dep `../..` (results below: v2.7.1) | our numbers |
| `mlkit/` | google_mlkit_face_mesh_detection | 0.5.0 | comparison anchor |
| `fdt/` | face_detection_tflite | 6.8.0 | reference |

There is no baseline app checked in; to measure size deltas,
`flutter create` a fresh template app with the same Flutter version and
identical assets.

Asset provenance and licenses: `assets/SOURCES.md`.

## Setup (once)

```
python3 bench/tool/prepare_assets.py
```

Extracts 300 frames from `assets/bench_face_10s.mp4` and copies
`portrait.jpg` and `frames/` into every app's `assets/` (all gitignored).

## Protocol

- Single image: full pipeline per call (detector + mesh every call, no
  cross-call state), 10 warmup runs, then 100 measured runs of per-call
  latency.
- Streaming: 300-frame sequence, detector on frame 0 only, internal ROI
  tracking after that (packages without tracking run the full pass every
  frame). First 30 frames discarded, steady-state stats + landmark jitter
  (mean per-landmark px displacement between frames).
- Stats: mean / median / p95 over the measured samples; published numbers
  quote mean.

## Run

```
cd bench/mine   # or mlkit / fdt
flutter test integration_test -d <device-id> | tee ../logs/mine-<device>.log
python3 bench/tool/aggregate.py bench/logs/*.log
```

Results print as `BENCH_JSON {...}` lines; the aggregate script turns all
logs into one markdown table. Windows desktop: `-d windows` (mine and fdt
only; the ML Kit mesh package is Android-only).

Published summary of these results: `doc/BENCHMARKS.md` (package doc,
linked from the main README). Keep it in sync when numbers change.

## Results: v2.7.1, Samsung SM-X930 (Dimensity 9400, Android 16)

Measured on v2.7.1 (log `mine-android-2.7.1.log`), default settings unless
noted. Mean ms over 100 runs after 10 warmups. Streaming cpu 468 is stable
across sessions (observed 1.46 to 1.54ms); the single-image numbers swing
with device thermal state (observed 2.6 to 4.5ms across sessions), so
treat streaming as the stable headline metric.

Single image (full detector + mesh per call):

| config | portrait (820x1024) |
| --- | --- |
| cpu, 468 mesh | 2.9 |
| cpu, attention 478 | 3.7 |
| xnnpack, 468 mesh | 2.9 |

Streaming (720p sequence, ROI tracking, steady state):

| config | per frame | jitter raw / OneEuro (px) |
| --- | --- | --- |
| cpu, 468 mesh | 1.54ms | 1.43 / 1.21 |
| cpu, attention 478 | 2.10ms | 1.51 / 1.28 |
| xnnpack, 468 mesh | 1.47ms | 1.43 / 1.21 |

Comparison anchor: `google_mlkit_face_mesh_detection` 0.5.0,
`FaceMeshDetectorOptions.faceMesh` (detection + 468 mesh in one call),
nv21 buffer prepared outside the measured call; log
`mlkit-meshdet-android.log`:

| config | portrait |
| --- | --- |
| single image, per call | 43.7 |
| streaming 720p, per frame | 49.6 (15/300 no mesh) |

Notes:
- cpu and xnnpack are within noise of each other (the bundled runtime
  applies XNNPACK in cpu mode too), which is why the matrix runs xnnpack
  in a single config.
- gpuV2 is deprecated and excluded from the matrix.

## App size

Windows PowerShell:

```
.\bench\tool\build_sizes.ps1
```

Results (release APK, android-arm64, identical bundled assets;
log `app-sizes-android.log`):

| app | apk MB | delta vs baseline MB |
| --- | --- | --- |
| baseline | 27.84 | 0 |
| mine | 47.34 | 19.5 |
| mlkit | 55.00 | 27.16 |

Deltas include each package's native libraries and bundled models. The
mlkit row was measured with `google_mlkit_face_detection`.

## Known caveats

- `flutter test integration_test` runs a DEBUG build. The plugin's
  CMakeLists forces -O3 for its native code regardless of build type,
  but Dart-side code still runs JIT in debug. Confirm headline numbers
  with a profile run before quoting them anywhere: `flutter drive
  --driver=test_driver/integration_test.dart
  --target=integration_test/bench_test.dart --profile` (release mode is
  not supported by Flutter Driver). The v2.7.1 numbers are confirmed this
  way (log `mine-android-2.7.1-profile.log`, streaming slightly faster).
- fdt input differs from ours: `detectFacesFromBytes` takes encoded JPEG,
  while our call takes pre-decoded RGBA. For a strict like-for-like run
  use their `detectFacesFromMatBytes` with raw pixels.
- GPU delegate results must check `activeDelegate` in the output JSON; a
  silent CPU fallback otherwise reads as a GPU number.
- minSdk: ML Kit and opencv_dart may need a higher Android minSdk than the
  Flutter template default; bump per-app `android/app/build.gradle.kts` if
  the build complains.
