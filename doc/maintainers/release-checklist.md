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

## CocoaPods path (macOS host)

The podspecs are only used by apps that turned Swift Package Manager off,
and nothing else builds them. CocoaPods support is removed in 4.0.0 (#45).
Until then, skip this section unless
`git diff v<last>..HEAD --stat -- ios macos` shows a change. A change whose
pull request already records a CocoaPods build of its final state counts as
checked.

`example/` is Swift Package Manager only and does not build with it turned
off, so use a throwaway app (or `bench/mine` with a generated iOS runner):

- [ ] `flutter create` an app in a temp directory and add this repository
      as a path dependency
- [ ] `flutter config --no-enable-swift-package-manager`
- [ ] in that app: `flutter build macos --release` and
      `flutter build ios --release --no-codesign`
- [ ] `flutter config --enable-swift-package-manager` (user-global setting,
      restore it even when a build failed)
- [ ] delete the temp directory

## After the release commit

- [ ] Tag `v<version>` on the release commit and publish a GitHub release
      with that version's CHANGELOG section as the notes (the `gh-release`
      skill does both)
