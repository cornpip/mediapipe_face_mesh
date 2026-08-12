# Running debug builds on a physical iOS device

`flutter run` hangs at `Installing and launching...` on an iOS 17+ device, with
no error message. Or the app is installed but dies the moment it starts.

Use **Flutter 3.38.0 or newer**. This is Flutter tooling behavior and is not
specific to this package — every Flutter iOS project hits it the same way.

## Why debug builds need a debugger

Flutter debug builds run Dart in JIT mode: the engine compiles Dart to machine
code while the app runs, then executes that freshly written memory. iOS forbids
executing writable memory, with one exception — a process carrying the
`get-task-allow` entitlement *while a debugger is attached*.

So on iOS, a debug build only runs with a debugger attached. Without one, the
Dart VM cannot allocate executable memory and the app aborts during
`Dart_Initialize`, before any Dart code (or any of this package's code) runs.
Release and profile builds are AOT-compiled ahead of time, need no debugger, and
are never affected.

## What changed in Flutter 3.38.0

Xcode 15 / iOS 17 replaced the old device connection stack with CoreDevice, and
the debug channel Flutter used before stopped working. Flutter's interim
solution was to drive the Xcode application over AppleScript to launch and debug
the app, which requires Xcode to be running with the project open and macOS
Automation permission granted to your terminal. When any of that does not line
up, the handshake never completes and `flutter run` waits forever.

Flutter 3.38.0 replaced that path with `devicectl` (install and launch) and LLDB
(debugging), so no Xcode automation is involved:

- [Use LLDB as the default debugging method for iOS 17+ and Xcode 26+](https://github.com/flutter/flutter/pull/173443)
- [Umbrella issue: use LLDB and devicectl on iOS 17+ physical devices](https://github.com/flutter/flutter/issues/173416)
- [Flutter 3.38.0 release notes](https://docs.flutter.dev/release/release-notes/release-notes-3.38.0)

## Symptoms and workarounds

| How the app is started | Debugger | Result |
| --- | --- | --- |
| `flutter run`, Flutter < 3.38.0 | not attached | hangs at `Installing and launching...` |
| `flutter run`, Flutter >= 3.38.0 | attached via LLDB | runs |
| Xcode, Run (⌘R) | attached | runs |
| Tapping the app icon on the device | none | aborts during Dart VM startup |
| `flutter run --release` / `--profile` | not needed | runs |

If you cannot move to Flutter 3.38.0 yet:

- open `ios/Runner.xcworkspace` in Xcode and run from there (⌘R), or
- keep that workspace open in Xcode and grant your terminal permission to control
  Xcode under System Settings → Privacy & Security → Automation, so the
  AppleScript path can complete, or
- use `flutter run --release` when you do not need the debugger
