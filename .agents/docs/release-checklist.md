# Release checklist

## Collect

- [ ] `git fetch --tags` (local tags can lag GitHub)
- [ ] Review `git log v<last>..HEAD --oneline` against the `-wip`
      section for missed bullets

## Version settings

- [ ] `pubspec.yaml` `version:` (drop the `-wip` suffix)
- [ ] `CHANGELOG.md` `## <version>` section (rename from `-wip`)
- [ ] `ios/mediapipe_face_mesh.podspec` and
      `macos/mediapipe_face_mesh.podspec` `s.version`
- [ ] `flutter pub get` in `example/` and `bench/mine/` (refresh locks)

## CocoaPods path

The podspecs are only used by apps that turned Swift Package Manager off,
so nothing else builds them. Build the example that way once, in a
separate worktree so the CocoaPods files it generates never touch this
checkout:

- [ ] `git worktree add /tmp/mpfm-pods HEAD`
- [ ] `flutter config --no-enable-swift-package-manager`
- [ ] in `/tmp/mpfm-pods/example`: `flutter build macos --release` and
      `flutter build ios --release --no-codesign`
- [ ] `flutter config --enable-swift-package-manager` (user-global setting)
- [ ] `git worktree remove --force /tmp/mpfm-pods`

## After the release commit

- [ ] Verify an annotated tag `v<version>` exists on the release commit
      (the GitHub release flow may create it)
