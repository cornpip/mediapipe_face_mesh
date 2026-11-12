# .agents/rules.md

Read this first. Read other docs only when the task touches that area.

- Repo conventions (commit prefixes, changelog `-wip` flow, engineering
  rules) are defined in `CONTRIBUTING.md` (repo root). Follow it.
- Work is tracked on Jira (project `MFM`). Branch names and commit
  subjects carry the card key (e.g. `MFM-27-readme-docs`) so GitHub for
  Jira links them to the card.

## Read-On-Demand

- Version bump / release: read `.agents/docs/release-checklist.md`.
- iOS or macOS bundled TensorFlow Lite binary replaced: read
  `.agents/docs/apple-binary-update.md`.
- `LICENSE`, `NOTICES`, or a bundled binary or model changed: read
  `.agents/docs/license-notices.md`.

## Agent-Specific

- Git actions that write history (commit, push, amend, tag, reset) require
  the user's explicit permission for that specific action, asked in the
  current exchange. Prior stated intent is not permission.
- Do not add AI attribution (a "Generated with ..." line or a
  `Co-Authored-By` AI trailer) to commit messages or PR descriptions.
- A `chore` commit's optional CHANGELOG bullet is the user's call: suggest
  one when it seems worth recording, never add it unprompted.
- Prose style: avoid em dashes in anything written for this repo (docs,
  CHANGELOG, code comments, commit messages). Use a comma or a separate
  sentence.
- When one of these docs disagrees with the code (a renamed file, a
  removed API), follow the code and fix the doc in the same change.
