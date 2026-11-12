# Windows desktop

- Chip and OS build: not recorded

Newest run first.

## 2.9.0-wip, 2026-09-01: delegates

Debug build, `multiface_profile_test`. The bundled dll's cpu path runs the
mesh models 4~5x slower than xnnpack (ms per pass):

| model | cpu | xnnpack |
| --- | --- | --- |
| v1 | 6.5 | 1.7 |
| attention | 8.1 | 2.3 |
| v2 | 19.0 | 3.6 |

The example and README recommend xnnpack there.
