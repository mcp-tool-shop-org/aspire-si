# aspire-si: how it works

Mapped at 2026-10-15 from commit 961a6dc by Atlas 1.24.0.

## What this is

ASPIRE training system (student, critic, teachers) with a training-dynamics export for ScalarScope (written by a person)

8 parts, mostly Python (246 files), shell (11), CSS (2), TypeScript (2), Astro (1) and JavaScript (1). Work enters through 4 doors; the busiest is CI, which reaches 5 parts. It publishes to PyPI. It deploys a site to GitHub Pages. People run aspire.

## What changed since 2026-10-15 (cfd95d3)

Nothing structural changed since 2026-10-15; 4 files added and 3 changed content.

## What comes in

1. **CI.** On a pull request to main or verifier-distill-prereg-draft touching 9 paths; on a push to main touching 7 paths; or by hand. Runs docs/runs/stage1-train/, examples/geometry_demo.py, packages/aspire-si-gates/tests/ and 78 more; checks aspire/ and integrations/.
2. **Release to PyPI.** When a release is published. Checks aspire/ and integrations/.
3. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
4. **aspire** (a command people run). Runs aspire/cli.py.

## What happens through CI

1. The workflow runs docs/runs/stage1-train/ in docs, examples/geometry_demo.py in examples, tests/ in tests and packages/aspire-si-gates/tests/; it checks aspire/ in aspire and integrations/ in integrations.
   1. Inside examples/geometry_demo.py, `main` does, in order: `GeometryRecorder` (aspire) and `record_step` (GeometryRecorder).
2. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Release to PyPI** checks aspire/ and integrations/, and publishes to PyPI.

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**aspire** (a command people run) runs aspire/cli.py.

## What breaks what

- **aspire** is imported by 1 part (examples), and by 1 more only from tests; it sits on the path of 3 doors.
- **integrations** is imported only from tests, by 1 part (tests), and sits on the path of 2 doors.
- **examples** is imported only from tests, by 1 part (tests), and sits on the path of 1 door.

## What tends to change together

- **packages/aspire-si-gates/src/aspire_si_gates/__init__.py** and **packages/aspire-si-gates/tests/test_package.py** changed together in 11 of 11 commits.
- **packages/aspire-si-gates/src/aspire_si_gates/__init__.py** and **packages/aspire-si-gates/src/aspire_si_gates/gates.py** changed together in 10 of 11 commits.
- **packages/aspire-si-gates/src/aspire_si_gates/gates.py** and **packages/aspire-si-gates/tests/test_package.py** changed together in 10 of 11 commits.
- **examples/sft-experiment/critic_heads.py** and **examples/sft-experiment/critic_heads_run.py** changed together in 8 of 9 commits, inside the examples part.
- **aspire/config.py** and **aspire/trainer.py** changed together in 4 of 7 commits, inside the aspire part.

1 file changed together with its own test, as expected.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 6 source files reach 10 revisions; the floor rises to 10 when 25 do.

## What no test touches

Every code part is imported by at least one test.

4 test files run in no workflow: docs/runs/stage1-eval/test_score.py, docs/runs/stage1-eval/test_score_found.py, docs/runs/stage1-eval/test_score_v2.py and 1 more.

## Written but never read

Every written place has a reader.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

- **docs/runs/stage1-training-data/** is written by docs (7 files).
- **docs/runs/stage1-training-data/pilot-t04.jsonl** has a block written by docs/runs/stage1-training-data/_fix_pilot_sources.py, docs/runs/stage1-training-data/_pilot_fixes.py and docs/runs/stage1-training-data/_pilot_fixes_r2.py.

## Hand-authored

People write .github/, the repository root and site/; 20 writes with paths built at run time may land here.

## Where to start

.github/workflows/ci.yml → examples/geometry_demo.py → aspire/geometry.py

Read those in order to follow one pull request end to end.

## What this map cannot see

- 3 import sites name a declared dependency that shares its name with a local module (openai); they are read as the dependency, which is not in this repository.
- 21 imports could not be resolved: `aspire/cli.py` imports a path built at run time; `docs/runs/stage1-train/apply.py` imports `student`, which is no module on its import path and no declared dependency; `docs/runs/stage1-train/batched.py` imports `harness`, which is no module on its import path and no declared dependency; and 18 more.
- 20 writes and 42 reads use paths built at run time and are not named here.
- 67 writes and 85 reads go to a path their caller passes, not to this repository.
- 1 read goes to the home directory (.cache/), not to this repository.
- 15 files belong to no part: packages/aspire-si-gates/README.md, packages/aspire-si-gates/pyproject.toml, packages/aspire-si-gates/src/aspire_si_gates/__init__.py and 12 more.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
