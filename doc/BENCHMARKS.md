# Benchmarks

- Device: Samsung SM-X930 (Dimensity 9400, Android 16)
- Measured:
  - `mediapipe_face_mesh`: 2.8.0
  - `google_mlkit_face_mesh_detection`: 0.5.0

## Method

- Single image: full pipeline per call (detector + mesh, no cross-call
  state), 10 warmup runs, then 100 measured runs. Values are mean ms.
- Inputs are pre-decoded buffers prepared outside the measured call, on
  both sides of every comparison (RGBA for this package, NV21 for ML Kit:
  each API's raw input format).
- Streaming: 300-frame 720p sequence, detector on frame 0 only, internal
  ROI tracking afterward. Frames are pre-decoded and calls issued
  back-to-back. First 30 frames discarded, steady-state stats.
- Jitter: mean per-landmark pixel displacement between consecutive frames.
- Tracking is validated per frame (landmark count and confidence, with
  failures cross-checked by an independent detector pass) and probed every
  30 frames against an independent detection. The clip fades to black over
  its last 15 frames; both packages see those as no-face frames and their
  latency stays in the samples.
- Harness: Flutter integration tests. The plugin's native code is compiled
  with `-O3` in every build type. The numbers quoted here are from a
  profile-mode (AOT) run.

## Single image

Full detector + mesh pass per call, mean ms. Test image: `portrait.jpg`
(820x1024, single face), Google's official MediaPipe test asset.

| config | per call |
| --- | --- |
| cpu, 468 mesh | 2.9 |
| cpu, attention 478 | 3.8 |
| cpu, FaceMesh-V2 478 | 5.2 |
| xnnpack, 468 mesh | 2.9 |
| xnnpack, FaceMesh-V2 478 | 5.2 |

Single-image latency swings with device thermal state (2.6 to 4.5 ms
observed across sessions); treat the streaming numbers below as the stable
metric.

## Streaming

720p sequence with ROI tracking, steady state (mean ms):

| config | per frame | jitter raw / OneEuro (px) |
| --- | --- | --- |
| cpu, 468 mesh | 1.46 ms | 1.43 / 1.21 |
| cpu, attention 478 | 2.08 ms | 1.51 / 1.28 |
| cpu, FaceMesh-V2 478 | 3.00 ms | 1.46 / 1.26 |
| xnnpack, 468 mesh | 1.47 ms | 1.43 / 1.21 |
| xnnpack, FaceMesh-V2 478 | 2.90 ms | 1.46 / 1.26 |

For scale: a 30 fps camera budgets 33 ms per frame and 60 fps budgets
16.7 ms, so these latencies leave nearly the whole frame to the app.

Note that these are sustained-throughput numbers: the continuous loop
keeps the CPU boosted. A real camera pipeline idles between frames, which
lets the governor drop clocks, so per-call latency in a live app can be
higher; how much depends on the device's power management and the app's
concurrent load. The benchmark harness can simulate a fixed camera
cadence to explore that scenario on a given device (pass
`--dart-define=PACED_FPS=<fps>` to the streaming suite in `bench/`).

ML Kit's face mesh on the same sequence runs at 49.6 ms per frame under
the same protocol, above the 33 ms budget of a 30 fps stream; see the
reference section below.

## ML Kit reference

Same device and inputs, `google_mlkit_face_mesh_detection` 0.5.0 in
`FaceMeshDetectorOptions.faceMesh` mode: one call runs face detection plus
the 468-point mesh, the same unit of work as this package's detector +
mesh pipeline. Input is an NV21 buffer prepared outside the measured call,
so no file IO or decode is inside the measured call.

| scenario | mean ms |
| --- | --- |
| single image, per call | 43.7 |
| streaming 720p, per frame | 49.6 |

Caveats for a fair reading:

- ML Kit's face mesh detection is beta and Android only; this comparison
  could not be run on iOS or Windows.
- It has no tracking mode, so the streaming number is a full pass per
  frame. 15 of 300 frames returned no mesh (the clip's fade-to-black tail,
  where this package's harness also reports no face); their latency is
  still included in the samples.

## App size

Adding this package to an empty Flutter app increases the release APK
(android-arm64, single ABI) by about 19.5 MB: the bundled TensorFlow Lite
C runtime plus the detector, mesh, attention, iris, and blendshapes
models. 2.8.0 adds the bundled FaceMesh-V2 model, about 2.4 MB on top of
that measured delta. The exact delta varies with build configuration (ABI splits, app
bundle delivery, shrinking). iOS has not been measured separately; App
Store thinning ships only the device slice of the bundled xcframework.

## Notes

- `cpu` and `xnnpack` are within noise of each other: the bundled runtime
  applies XNNPACK in cpu mode too.
- FaceMesh-V2 lets XNNPACK delegate its whole graph in one partition (471
  of 471 nodes; the attention model's custom ops split its graph), but its
  256x256 input carries 1.8x the pixels of the attention model's 192x192,
  and on this device it lands about 40% slower than attention. Its raw
  jitter is slightly lower (1.46 vs 1.51 px). FaceMesh-V2 is the
  recommended model on accuracy grounds (see below); attention is the
  lowest-latency 478 option.
- `gpuV2` is deprecated and excluded; in our runs the GPU delegate was
  slower for these models and only added binary size.

## Accuracy (upstream model cards)

Google evaluates both models on the same internal protocol (1,700
smartphone selfies across 17 geographic subregions, mean absolute error
normalized by interocular distance, human-annotator baseline 2.56%).
Tracking-mode results: attention mesh 3.28% IOD MAE, FaceMesh-V2 2.62%,
i.e. FaceMesh-V2 cuts the error by about 20% and reaches
human-annotator-level accuracy. This is Google's own evaluation set, not
an independent benchmark. Sources: the
[Attention Mesh model card](https://mediapipe.page.link/attentionmesh-mc)
and the
[FaceMesh-V2 model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf).
