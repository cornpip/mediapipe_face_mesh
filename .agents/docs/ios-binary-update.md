# Apple bundled binary update

When a bundled iOS or macOS framework binary is replaced:

- iOS: re-sync that framework's `Headers/` from `src/include`, in every
  xcframework slice. iOS compiles against the bundled headers, and stale
  ones silently disagree with the binary about struct layout.
- macOS: the xcframework carries the same `Headers/` for consistency only.
  macOS compiles against `src/include` directly, so stale headers there are
  harmless, but re-sync them with the iOS ones.
