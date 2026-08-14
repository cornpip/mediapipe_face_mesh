# iOS bundled binary update

When a bundled iOS framework binary is replaced:

- Re-sync that framework's `Headers/` from `src/include`, in every
  xcframework slice. iOS compiles against the bundled headers, and stale
  ones silently disagree with the binary about struct layout.
