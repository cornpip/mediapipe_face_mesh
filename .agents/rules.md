# .agents/rules.md

Read this first. Read other docs only when the task touches that area.

- Repo conventions (commit prefixes, changelog `-wip` flow, engineering
  rules) are defined in `CONTRIBUTING.md` (repo root). Follow it.
- Work is tracked with GitHub issues, grouped by release in milestones
  (e.g. `3.3.0`). An issue is only open or closed, there are no status
  columns. When a change has an issue, the branch name starts with its
  number (e.g. `36-tflite-interpreter`), the PR description says
  `Closes #<n>`, and the PR goes in the same milestone, so merging closes
  the issue. Commit subjects carry no issue number.

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
