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

Extracts 300 frames from `assets/bench_face_10s.mp4`, records the source
video fps in `frames/meta.json`, and copies `portrait.jpg` and `frames/`
into every app's `assets/` (all gitignored). Re-run it on older checkouts:
the streaming suites now require `frames/meta.json`.

## Protocol

- Single image: full pipeline per call (detector + mesh every call, no
  cross-call state), 10 warmup runs, then 100 measured runs of per-call
  latency.
- Streaming (back-to-back, `streaming` suite): 300-frame sequence,
  detector on frame 0 only, internal ROI tracking after that (packages
  without tracking run the full pass every frame). Frames are decoded up
  front so the measured loop touches nothing but the inference call;
  per-frame decode between calls pollutes caches and roughly doubled the
  measured latency in an A/B run. First 30 frames discarded,
  steady-state stats +
  landmark jitter (mean per-landmark px displacement between frames).
- Streaming (paced, `streaming_paced` suite, opt-in): only runs when a
  cadence is given via `--dart-define=PACED_FPS=<fps>`. Same sequence,
  but each frame is decoded in the loop (outside the stopwatch, standing
  in for camera frame delivery) and processed on its wall-clock deadline
  at that fps. Idle gaps let the CPU governor drop clocks, so this
  measures device DVFS behavior under the chosen cadence as much as the
  package; treat it as an exploratory scenario test, not a tracked
  metric. The back-to-back number is the boosted best case.
- Validation (mine only): every frame must return the expected landmark
  count with score > 0.5. Frames that fail are cross-checked with an
  independent detector pass: genuine no-face frames are counted as
  `noFaceFrames` (the clip fades to black over its last 15 frames, so 15
  is the expected value), while `trackingFailFrames` fails the test. Every
  30th frame an independent detection is compared against the tracked
  landmark bbox (`roiDriftIou*`), so a frozen or drifting tracker cannot
  hide behind a good jitter number.
- Thermal: every measured config starts with a 30 s idle cooldown
  (`kCooldownSeconds`) so later configs are not measured on a hotter
  device than earlier ones.
- Stats: mean / median / p95 over the measured samples; published numbers
  quote mean. For paced runs read median/p95 instead; their absolute
  values swing with session thermal state.

## Run

```
cd bench/mine   # or mlkit / fdt
flutter test integration_test -d <device-id> | tee ../logs/mine-<device>.log
python3 bench/tool/aggregate.py bench/logs/*.log
```

Results print as `BENCH_JSON {...}` lines; the aggregate script turns all
logs into one markdown table. Windows desktop: `-d windows` (mine and fdt
only; the ML Kit mesh package is Android-only).

Optional camera-cadence scenario (mine only, see Protocol):

```
flutter test integration_test/bench_test.dart -d <device-id> --dart-define=PACED_FPS=30
```

Published summary of these results: `doc/BENCHMARKS.md` (package doc,
linked from the main README). Keep it in sync when numbers change.

## Results: v2.7.1 (reworked harness), Samsung SM-X930 (Dimensity 9400, Android 16)

Measured 2026-08-14 on the reworked harness, profile build (log
`mine-android-rework-profile.log`; debug run `mine-android-rework2.log`),
default settings unless noted. Streaming back-to-back reproduces the
numbers published for the previous harness (1.54 / 2.10 / 1.47), so the
rework did not shift the headline metric. Single-image numbers swing with
device thermal state (observed 2.6 to 4.5ms across sessions); treat
streaming back-to-back as the stable headline metric.

Single image (full detector + mesh per call, mean ms):

| config | portrait (820x1024) |
| --- | --- |
| cpu, 468 mesh | 3.3 |
| cpu, attention 478 | 4.0 |
| xnnpack, 468 mesh | 3.0 |

Streaming back-to-back (720p sequence, ROI tracking, steady state, mean
ms):

| config | per frame | jitter raw / OneEuro (px) |
| --- | --- | --- |
| cpu, 468 mesh | 1.48 | 1.43 / 1.21 |
| cpu, attention 478 | 2.12 | 1.51 / 1.28 |
| xnnpack, 468 mesh | 1.50 | 1.43 / 1.21 |

Tracking validation in these runs: `trackingFailFrames=0` in every
config, drift IoU 0.77 to 0.80, `noFaceFrames=15` (the black tail),
`scoreMin=1.00` on face-bearing frames.

Comparison anchor: `google_mlkit_face_mesh_detection` 0.5.0,
`FaceMeshDetectorOptions.faceMesh` (detection + 468 mesh in one call),
nv21 buffer prepared outside the measured call; log
`mlkit-meshdet-android.log`:

| config | portrait |
| --- | --- |
| single image, per call | 43.7 |
| streaming 720p, per frame | 49.6 (15/300 no mesh: the black tail frames) |

Notes:
- cpu and xnnpack are within noise of each other (the bundled runtime
  applies XNNPACK in cpu mode too), which is why the matrix runs xnnpack
  in a single config.
- gpuV2 is deprecated and excluded from the matrix.

### FaceMesh-V2 (2.8.0-wip, measured 2026-08-23)

Profile run, log `mine-android-v2-profile.log`. The matrix key changed
from `attention: bool` to `model` (v1 / attention / v2, logged as
base/faceMeshV2 in this run before the enum rename); xnnpack now runs v1
(cpu parity) and v2 (full-graph delegation check).

| config | single image | streaming | jitter raw / OneEuro (px) |
| --- | --- | --- | --- |
| cpu, faceMeshV2 | 5.2 | 3.00 | 1.46 / 1.26 |
| xnnpack, faceMeshV2 | 5.2 | 2.90 | 1.46 / 1.26 |

XNNPACK delegates the whole FaceMesh-V2 graph (471 of 471 nodes, one
partition, confirmed in the run log), but the 256x256 input (1.8x the
pixels of attention's 192x192) still leaves it about 40% slower than
attention on this device, with slightly lower raw jitter (1.46 vs
1.51 px). Tracking validation passed in every config. The pre-existing
configs reproduced the v2.7.1 numbers within noise (base 1.46/1.47,
attention 2.08).

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
  not supported by Flutter Driver). The results above are from such a
  profile run. `oneEuroCostMs` is pure Dart and is the value most
  distorted by debug JIT (roughly 2x: 0.14 debug vs 0.07 profile
  observed).
- The back-to-back suite preloads all decoded frames, about 1.1 GB of RAM
  at 720p. On low-RAM devices run the paced suite instead
  (`--dart-define=PACED_FPS=<fps>` with `--plain-name streaming_paced`),
  which decodes one frame at a time.
- Close other apps on the device before measuring; a resident background
  app was observed in one session's logs and may add variance.
- fdt input differs from ours: `detectFacesFromBytes` takes encoded JPEG,
  while our call takes pre-decoded RGBA. For a strict like-for-like run
  use their `detectFacesFromMatBytes` with raw pixels.
- GPU delegate results must check `activeDelegate` in the output JSON; a
  silent CPU fallback otherwise reads as a GPU number.
- minSdk: ML Kit and opencv_dart may need a higher Android minSdk than the
  Flutter template default; bump per-app `android/app/build.gradle.kts` if
  the build complains.
