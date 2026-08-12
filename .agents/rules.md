# .agents/rules.md

Read this first. Read other docs only when the task touches that area.

- Follow `SECRET_AGENTS_RULE.md` (repo root) if it exists.

## Read-On-Demand

- Version bump / release: read `.agents/docs/release-checklist.md`.
- `CHANGELOG.md` entry: read `.agents/docs/changelog-style.md`.

## Always Apply

- Published production package: never break the public API or change default
  behavior. New capabilities are opt-in (default off).
- Native declarations changed: regenerate bindings with
  `dart run ffigen --config ffigen.yaml`.
- iOS bundled framework binary replaced: re-sync that framework's `Headers/`
  from `src/include`, in every xcframework slice. iOS compiles against those
  bundled headers, and stale ones silently disagree with the binary about
  struct layout.
- Prose style: avoid em dashes (`—`) in anything written for this repo (docs,
  CHANGELOG, code comments, commit messages). Use a comma, colon, semicolon, or
  a separate sentence instead.
