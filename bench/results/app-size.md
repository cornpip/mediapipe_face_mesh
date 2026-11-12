# App size

Not tied to a device. Release APK, android-arm64, identical bundled
assets, log `app-sizes-android.log`.

| app | apk MB | delta vs baseline MB |
| --- | --- | --- |
| baseline | 27.84 | 0 |
| mine | 47.34 | 19.5 |
| mlkit | 55.00 | 27.16 |

Deltas include each package's native libraries and bundled models. The
mlkit row was measured with `google_mlkit_face_detection`.
