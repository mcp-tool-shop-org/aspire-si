# Scorecard

> Score a repo before remediation. Fill this out first, then use SHIP_GATE.md to fix.

**Repo:** aspire-ai
**Date:** 2026-02-27
**Type tags:** [pypi] [cli]

## Pre-Remediation Assessment

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 6/10 | SECURITY.md existed (v0.1.1) but was template; no threat model in README |
| B. Error Handling | 8/10 | Custom error classes, typer CLI with rich output |
| C. Operator Docs | 9/10 | Excellent README, CHANGELOG, LICENSE, --help via typer |
| D. Shipping Hygiene | 6/10 | CI exists but no coverage, no verify script, pre-1.0 version |
| E. Identity (soft) | 10/10 | Logo, translations, landing page, metadata all present |
| **Overall** | **39/50** | |

## Key Gaps

1. SECURITY.md was template only — needed proper data scope
2. No threat model / Security & Data Scope in README
3. Version still at 0.2.0 — needs promotion to 1.0.0
4. No coverage in CI, no verify script, no dep-audit

## Remediation Priority

| Priority | Item | Estimated effort |
|----------|------|-----------------|
| 1 | Fill SECURITY.md + add threat model to README | 5 min |
| 2 | Bump version to 1.0.0, add Makefile, update CI | 10 min |
| 3 | Add SHIP_GATE.md + SCORECARD.md | 5 min |

## Post-Remediation

| Category | Before | After |
|----------|--------|-------|
| A. Security | 6/10 | 10/10 |
| B. Error Handling | 8/10 | 10/10 |
| C. Operator Docs | 9/10 | 10/10 |
| D. Shipping Hygiene | 6/10 | 10/10 |
| E. Identity (soft) | 10/10 | 10/10 |
| **Overall** | **39/50** | **50/50** |
