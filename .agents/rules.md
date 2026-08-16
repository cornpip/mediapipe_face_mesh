# .agents/rules.md

Read this first. Read other docs only when the task touches that area.

- Follow `SECRET_AGENTS_RULE.md` (repo root) if it exists.
- Repo conventions (commit prefixes, changelog `-wip` flow, engineering
  rules) are defined in `CONTRIBUTING.md` (repo root); follow it.

## Read-On-Demand

- Version bump / release: read `.agents/docs/release-checklist.md`.
- `CHANGELOG.md` entry: read `.agents/docs/changelog-style.md`.
- iOS bundled framework binary replaced: read
  `.agents/docs/ios-binary-update.md`.
- `LICENSE`, `NOTICES`, or a bundled binary or model changed: read
  `.agents/docs/license-notices.md`.

## Agent-Specific

- Git actions that write history (commit, push, amend, tag, reset) require
  the user's explicit permission for that specific action, asked in the
  current exchange. Prior stated intent is not permission.
- A `chore` commit's optional CHANGELOG bullet is the user's call: suggest
  one when it seems worth recording, never add it unprompted.
- Prose style: avoid em dashes in anything written for this repo (docs,
  CHANGELOG, code comments, commit messages); use a comma, colon,
  semicolon, or a separate sentence.
