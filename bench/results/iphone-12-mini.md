# iPhone 12 mini

- Chip: A14 Bionic (no Arm SME)
- OS: iOS 26.6

Newest run first. Values are median ms unless noted.

## 3.2.3-wip, 2026-11-12: runtime without SME and KleidiAI kernels

Profile runs of the full matrix, one per runtime, on the same build
otherwise. Logs `mine-ios-iphone12mini-old.log` (runtime bundled in 3.2.2)
and `mine-ios-iphone12mini-nosme.log` (runtime rebuilt without XNNPACK's
Arm SME, SME2, and KleidiAI kernels). The installed app binaries were
disassembled to confirm which runtime each run used (38 `smstart` vs 0).

`bench/mine` has no iOS runner checked in. One was generated for these
runs with `flutter create --platforms=ios .` and not committed.

Single image (full detector + mesh per call):

| config | 3.2.2 runtime | without SME | delta |
| --- | --- | --- | --- |
| cpu, v1 | 3.49 | 3.51 | +0.6% |
| cpu, attention | 4.12 | 4.16 | +0.8% |
| cpu, v2 | 6.52 | 6.58 | +1.0% |
| xnnpack, v1 | 3.52 | 3.50 | -0.6% |
| xnnpack, v2 | 6.50 | 6.58 | +1.3% |

Streaming back-to-back (720p sequence, ROI tracking, steady state):

| config | 3.2.2 runtime | without SME | delta | jitter raw / OneEuro (px) |
| --- | --- | --- | --- | --- |
| cpu, v1 | 2.38 | 2.45 | +2.7% | 1.44 / 1.22 |
| cpu, attention | 2.61 | 2.58 | -1.2% | 1.53 / 1.29 |
| cpu, v2 | 5.08 | 5.05 | -0.6% | 1.46 / 1.25 |
| xnnpack, v1 | 2.00 | 1.95 | -2.4% | 1.44 / 1.22 |
| xnnpack, v2 | 5.04 | 5.03 | -0.2% | 1.46 / 1.25 |

Jitter, drift IoU (0.80), `scoreMin`, `trackingFailFrames=0`, and
`noFaceFrames=15` are identical between the two runtimes in every config,
so removing the kernels did not change inference output.

Notes:
- medians are quoted because the 3.2.2 run of streaming cpu v1 had one
  49.6ms frame that pulled its mean to 3.17 (median 2.38, without SME
  mean 2.53).
- this device has no SME, so neither runtime selects an SME kernel here.
  The runs show that the rebuilt runtime works and is not slower on
  pre-A18 hardware. They do not measure the cost of losing SME on iPhone
  16 and 17, where the 3.2.2 runtime crashes.
- one run per runtime, one device.
