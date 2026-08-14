# Release checklist

## Collect

- [ ] `git fetch --tags` (local tags can lag GitHub)
- [ ] Review `git log v<last>..HEAD --oneline` against the `-wip`
      section for missed bullets

## Version settings

- [ ] `pubspec.yaml` `version:` (drop the `-wip` suffix)
- [ ] `CHANGELOG.md` `## <version>` section (rename from `-wip`; bullets
      only)
- [ ] `ios/mediapipe_face_mesh.podspec` `s.version` (easy to miss)
- [ ] `flutter pub get` in `example/` (refresh lock)

## After the release commit

- [ ] Verify an annotated tag `v<version>` exists on the release commit
      (the GitHub release flow may create it)
