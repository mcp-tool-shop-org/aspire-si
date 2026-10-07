# Ship Gate

> No repo is "done" until every applicable line is checked.
> Copy this into your repo root. Check items off per-release.

**Tags:** `[all]` every repo · `[npm]` `[pypi]` `[vsix]` `[desktop]` `[container]` published artifacts · `[mcp]` MCP servers · `[cli]` CLI tools

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (report email, supported versions, response timeline) (2026-02-27)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) (2026-02-27)
- [x] `[all]` No secrets, tokens, or credentials in source or diagnostics output (2026-02-27) — API keys read from env vars only
- [x] `[all]` No telemetry by default — state it explicitly even if obvious (2026-10-06) — `use_wandb` defaults to off; nothing is sent unless a teacher API is configured

### Default safety posture

- [x] `[cli|mcp|desktop]` Dangerous actions (kill, delete, restart) require explicit `--allow-*` flag (2026-02-27) — CLI is read/write to user-specified dirs only, no destructive ops
- [x] `[cli|mcp|desktop]` File operations constrained to known directories (2026-02-27) — training data and checkpoint dirs only
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server

## B. Error Handling

- [x] `[all]` Errors follow the Structured Error Shape: `code`, `message`, `hint`, `cause?`, `retryable?` (2026-10-06) — `aspire.errors.AspireError`; teacher, config and input errors subclass it
- [x] `[cli]` Exit codes: 0 ok · 1 user error · 2 runtime error · 3 partial success (2026-10-06) — `aspire.cli.run`: 1 for AspireError user errors, 2 for runtime, 130 on interrupt; no partial-success case exists
- [x] `[cli]` No raw stack traces without `--debug` (2026-10-06) — `run()` prints code/message/hint; `--debug` re-raises (tests/test_cli_errors.py)
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[desktop]` SKIP: not a desktop application
- [ ] `[vscode]` SKIP: not a VS Code extension

## C. Operator Docs

- [x] `[all]` README is current: what it does, install, usage, supported platforms + runtime versions (2026-02-27)
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) (2026-02-27)
- [x] `[all]` LICENSE file present and repo states support status (2026-02-27)
- [x] `[cli]` `--help` output accurate for all commands and flags (2026-02-27) — typer auto-generates help
- [x] `[cli|mcp|desktop]` Logging levels defined: silent / normal / verbose / debug — secrets redacted at all levels (2026-10-06) — `--quiet` / default / `--verbose` / `--debug`; keys live only in env vars and are never printed
- [ ] `[mcp]` SKIP: not an MCP server
- [x] `[complex]` HANDBOOK (2026-10-06) — Starlight handbook, 8 pages, at /aspire-si/handbook/

## C. Operator Docs

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists (test + build + smoke in one command) (2026-02-27) — Makefile verify target
- [ ] `[all]` SKIP: no release tag yet in the restored repository; the first aspire-si tag is cut from the version in pyproject.toml
- [x] `[all]` Dependency scanning runs in CI (ecosystem-appropriate) (2026-02-27) — dep-audit job
- [x] `[all]` Automated dependency update mechanism exists (2026-02-27)
- [ ] `[npm]` SKIP: not an npm package
- [x] `[npm]` `engines.node` set · `[pypi]` `python_requires` set (2026-02-27) — >=3.10
- [x] `[npm]` Lockfile committed · `[pypi]` Clean wheel + sdist build (2026-02-27) — hatchling build, twine check in CI
- [ ] `[vsix]` SKIP: not a VS Code extension
- [ ] `[desktop]` SKIP: not a desktop application

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header (2026-10-06) — the artwork still reads Aspire.AI (brand slug aspire-ai); a renamed logo is pending
- [x] `[all]` Translations (polyglot-mcp, 8 languages) (2026-02-27)
- [x] `[org]` Landing page (@mcptoolshop/site-theme) (2026-02-27)
- [x] `[all]` GitHub repo metadata: description, homepage, topics (2026-02-27)

---

## Gate Rules

**Hard gate (A–D):** Must pass before any version is tagged or published.
If a section doesn't apply, mark `SKIP:` with justification — don't leave it unchecked.

**Soft gate (E):** Should be done. Product ships without it, but isn't "whole."

**Checking off:**
```
- [x] `[all]` SECURITY.md exists (2026-02-27)
```

**Skipping:**
```
- [ ] `[pypi]` SKIP: not a Python project
```
