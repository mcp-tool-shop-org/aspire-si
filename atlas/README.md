# aspire-si: how it works

Mapped at 2026-10-06 from commit 478093f by Atlas 1.24.0.

## What this is

ASPIRE training system (student, critic, teachers) with a training-dynamics export for ScalarScope (written by a person)

8 parts, mostly Python (93 files), CSS (2), TypeScript (2), Astro (1) and JavaScript (1). Work enters through 4 doors; the busiest is CI, which reaches 3 parts. It publishes to PyPI. It deploys a site to GitHub Pages. People run aspire.

## What changed since the last map

This is the first map.

## What comes in

1. **CI.** On a pull request to main touching 7 paths; on a push to main touching 7 paths; or by hand. Runs tests/; checks aspire/ and integrations/.
2. **Publish to PyPI.** When a release is published. Checks aspire/ and integrations/.
3. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
4. **aspire** (a command people run). Runs aspire/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests; it checks aspire/ in aspire and integrations/ in integrations.
2. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Publish to PyPI** checks aspire/ and integrations/, and publishes to PyPI.

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**aspire** (a command people run) runs aspire/cli.py.

## What breaks what

- **aspire** is imported by 1 part (examples), and by 1 more only from tests; it sits on the path of 3 doors.
- **integrations** is imported only from tests, by 1 part (tests), and sits on the path of 2 doors.

## What tends to change together

No two source files changed together often enough to name.

Window: 180 days; a pair counts from 3 shared commits, since the window holds fewer than 30 qualifying commits.

## What no test touches

- **examples** is imported by no test.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, docs/, the repository root and site/; 3 writes with paths built at run time may land here.

## Where to start

aspire/cli.py → aspire/__init__.py → aspire/trainer.py → aspire/config.py

Read those in order to follow one run of aspire end to end. This path follows aspire (a command people run) from its entry, since CI runs only tests and checks.

## What this map cannot see

- 3 import sites name a declared dependency that shares its name with a local module (openai); they are read as the dependency, which is not in this repository.
- 14 imports could not be resolved: `aspire/cli.py` imports a path built at run time; `aspire/trainer.py` imports `aspire.geometry`, which is no module on its import path and no declared dependency; `integrations/code/examples/basic_critique.py` imports `aspire.integrations.code`, which is no module on its import path and no declared dependency; and 11 more.
- 3 writes and 8 reads use paths built at run time and are not named here.
- 4 writes and 18 reads go to a path their caller passes, not to this repository.
- 1 read goes to the home directory (.cache/), not to this repository.
- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 20 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
