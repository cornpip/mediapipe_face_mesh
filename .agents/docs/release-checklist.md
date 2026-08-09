# Release checklist

## Version settings

- [ ] `pubspec.yaml` `version:`
- [ ] `CHANGELOG.md` `## <version>` section (bullets only)
- [ ] `ios/mediapipe_face_mesh.podspec` `s.version` (easy to miss)
- [ ] `flutter pub get` in `example/` (refresh lock)