# Changelog style

Rules for `CHANGELOG.md` entries.

## Structure

- One `## <version>` section per release, newest first. No headings inside a
  section except an optional `### Migrating from <version>` block.
- Flat `- ` bullets, no Added/Changed/Fixed subheadings. Details and platform
  caveats of a single feature go in nested sub-bullets under that feature.

## Bullets

- Start with a lowercase verb naming the change kind: `add`, `change`,
  `fix`, `improve`, ... Scope prefixes for non-package changes: `docs:`,
  `example:`.
- Name public symbols in backticks (`startPreviewAuto()`).

## Classification

Ask: "did something a previous user already had become different?"

- No, it is new: `add`. Limits of the new thing are its sub-bullets, never
  separate entries.
- Yes, and their code must change: `**BREAKING**:` with migration steps as
  sub-bullets (use a `### Migrating from <version>` section only when a
  release has several breaking items).
- Yes, but no code change needed: plain `change`/`improve`, even when
  user-visible.
- It was broken and now works as intended: `fix`.
