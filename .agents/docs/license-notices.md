# License and notices

Attributions and what each bundled component is: `THIRD_PARTY_NOTICES.md`.

Flutter's collector reads `NOTICES` in preference to `LICENSE`, and the
Apache-2.0 section 4(b) notice that the bundled runtime binaries are rebuilt
has to reach a consuming app. It lives in `NOTICES` rather than `LICENSE`
because prose in `LICENSE` costs pub.dev's license detection; pana's
`detectLicenseInContent` is the authority on the limit. Two stacked license
texts would detect fine in `LICENSE`; the prose is what breaks it.

The `NOTICES` multi-license format is Flutter's, not ours. Follow the
`LicenseCollector` doc comment in
`packages/flutter_tools/lib/src/license_collector.dart` in the installed SDK.

Keep these in sync:

- `NOTICES` holds verbatim copies of `LICENSE` and `LICENSE-APACHE-2.0.txt`
  with the attribution prose between them. Re-sync the copies whenever either
  source file changes.
- A new or rebuilt bundled binary or model updates both
  `THIRD_PARTY_NOTICES.md` and the `NOTICES` attribution prose.

Verify: `flutter build bundle` in `example/`, then gunzip
`example/build/flutter_assets/NOTICES.Z` and confirm every block of `NOTICES`
appears there as its own entry.
