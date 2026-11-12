# License and notices

Attributions and what each bundled component is: `THIRD_PARTY_NOTICES.md`.

- Never add prose to `LICENSE`. It breaks pub.dev license detection.
  Attribution prose, including the Apache-2.0 4(b) notice that the runtime
  binaries are rebuilt, lives in `NOTICES`.
- `NOTICES` uses Flutter's multi-license format (see the `LicenseCollector`
  doc comment in `flutter_tools/lib/src/license_collector.dart`). It holds
  verbatim copies of `LICENSE` and `LICENSE-APACHE-2.0.txt` with the
  attribution prose between them. Re-sync the copies whenever either source
  file changes.
- A new or rebuilt bundled binary or model updates both
  `THIRD_PARTY_NOTICES.md` and the `NOTICES` attribution prose.

Verify: `flutter build bundle` in `example/`, then gunzip
`example/build/flutter_assets/NOTICES.Z` and confirm every block of `NOTICES`
appears there as its own entry.
