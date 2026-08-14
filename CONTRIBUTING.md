# Contributing

## Commits

Commit subjects start with a type prefix:

- `feat` / `fix` / `perf` / `change`: package-visible changes
- `docs` / `example`: shipped non-code changes (docs and the example app
  are part of the published package)
- `chore`: repo-internal work (tooling, bench harness, agent rules)

Split mixed commits so each part keeps its prefix.

## Changelog

The topmost CHANGELOG section is `## <version>-wip`; it accumulates
bullets for the next release.

- Every non-`chore` commit adds its bullet to that section in the same
  commit. If the section does not exist yet, open it and set pubspec
  `version:` to the same `-wip` value in that commit.
- Pick the smallest bump the accumulated changes justify (docs or fix:
  patch); rename the section heading and pubspec when a later change
  needs a bigger bump.
- `chore` commits add no bullet by default; include one when it is worth
  recording.
- Bullet style: `.agents/docs/changelog-style.md`.

## Engineering rules

- New capabilities are opt-in (default off); default-behavior changes
  wait for a major release.
- When native declarations change, regenerate bindings with
  `dart run ffigen --config ffigen.yaml`.

## Tests and benchmarks

- Package tests: `flutter test` at the repo root.
- Benchmarks: protocol and how to run in `bench/README.md`.
