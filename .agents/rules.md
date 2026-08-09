# .agents/rules.md

Read this first. Read other docs only when the task touches that area.

- Follow `SECRET_AGENTS_RULE.md` (repo root) if it exists.

## Read-On-Demand

- Version bump / release: read `.agents/docs/release-checklist.md`.

## Always Apply

- Published production package: never break the public API or change default
  behavior. New capabilities are opt-in (default off).
- Native declarations changed: regenerate bindings with
  `dart run ffigen --config ffigen.yaml`.
