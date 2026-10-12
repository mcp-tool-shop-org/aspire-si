# Stage 1 sealed task set: frozen

**Frozen 2026-10-10.** The freeze came after three reviews:
- R&D's key check, complete at this hash;
- the Publisher's read of 15 per category;
- the maintainer's approval of a 70-item random sample.

**The set is sealed.** Its tasks never appear in this repository, in any lesson, quiz or training data, or in
anything derived from them. It lives only in the run directory and on the backup drive, read-only. This file
records its hashes so every later sitting can prove it scored exactly this set.

- **Size and mix:** 130 tasks: 6 categories × 20, plus 10 correction tasks. Keys: 40 supported, 55
  unsupported, 35 cannot_tell. 15 are escalate-keyed, all for medical, legal, safety or money advice.
- **Sources:** real text at pinned commits from portlight, ai-eyes-mcp, backpropagate, ai-rpg-engine,
  vocal-synth-engine, prompt-craft and offrig (README). The pinned sources are hashed below.
- **Scoring:** [`stage1-eval/score.py`](stage1-eval/score.py) (pinned in the baseline run note), at seeds 0, 1
  and 2, at the baseline, after round 1, and at the final round only.
- **Combined build-script hash:** `16b9f6945c3fe92bc11260bf7d39ccc3cf2c88548068e98eec31c8063ba1d431`. It's the
  sha256 of the build scripts concatenated in this order: sealed_build.py, sealed_common.py,
  sealed_tasks_a.py–g.py, sealed_tasks_k.py, taskset_trees.py.

```text
set:
e1e515b4201b3341c363d0cf1061356acf7775c3d4b5b5194afa9790223fc683 *taskset-sealed.jsonl
a71ebffcf01b4584addbe08826a4dc5570383a31c5eda1c184754ee5321bd2fe *taskset-sealed.md

build scripts:
15f6e762c56f8631160a40793cc9136c92e0d5bca2aa1b92e0688a2b91eb317c *sealed_build.py
218d5da3a3cac78241605f28a4e2f96b9bce88091684f9e13927a80373a5ddd5 *sealed_common.py
bb61a22d1a874c811c74c405d2a634d78d1ae47221b7a1631e9b0fba021058c5 *sealed_tasks_a.py
37754d06d00b400edaa649d12ac7536fff6e9396de150c97af8f2e0bfc2a4178 *sealed_tasks_b.py
d2dbfbda86a0d8e6467b3b70043888f16d27428f23f38d340a487be5713438e3 *sealed_tasks_c.py
25648ce3b7f70b770f793485c9ee2e8e01acfccd4130a6e1b813dd62adbb00c9 *sealed_tasks_d.py
caaf2185a30c9a40055081840f97d97beeefdcba6387d73d8c585869ba3f5fb0 *sealed_tasks_e.py
29b8f614daa46c8d41eef1910aa722604e4f0728366b311a08f4d6442c33d5ea *sealed_tasks_f.py
ce204bcf4cd27c57b85eca1edb262d49011af2c7589e345bf61f9360b4a2c2d3 *sealed_tasks_g.py
9ae17412b0ecbae5a5f5e7ae5d9766885bc520f96cead9888bd6ff4f7312f90e *sealed_tasks_k.py
5e3c33c763304b45f069dd16f3cc7c7a7fccf7c94a49232031fa3e9ac0398f10 *taskset_trees.py

pinned sources:
74dde46b62ed96bc31a9d5a1a2d4a4e6600b0655c6c724efe23c23ea7c4d664e *realsrc2/PINS.txt
5ec4e9e70e5bc59760cf01253b25d8e457d1aa9c21b7dfdb43fb8fc8e1951072 *realsrc2/ai-eyes-mcp--CHANGELOG.md
6520f2e70b3ee9cfffc41f75b520f7ef6e41326dad521b432373d3899bf94167 *realsrc2/ai-eyes-mcp--README.md
ba9945a4b369ba4d350036509580a44b6e50dc32d3f85277de238019766b5765 *realsrc2/ai-rpg-engine--CHANGELOG.md
614be8af7b7e4f7d27823cc62b811b7b8893caca8d7cb5bdb6e907418f4f628b *realsrc2/ai-rpg-engine--README.md
93444d3b705315094a42a06592db728d27bb3c2719add5bcf1f1092478456de2 *realsrc2/backpropagate--CHANGELOG.md
d514d786ec38b71420df47514560ced15bafe77fa198b4f1f05df9c02b272057 *realsrc2/backpropagate--README.md
a6137c0254a58c31ce14e3976a81d623ebe0dff3c9707e70129d8bde743a3f07 *realsrc2/loadout-os--CHANGELOG.md
367e5141e4f5884e5791c4021f33a9faa958840a5decf44e1b273a87aad1c459 *realsrc2/loadout-os--LICENSE
f292b7aaaf07caee8f9add73d1229cf60f4fad0259eb823d6c89ad1d6c18f781 *realsrc2/loadout-os--README.md
a624f10648680e3e2e194c67c6e093270cfec769d0179a02cbca5bc20c641ed7 *realsrc2/portlight--CHANGELOG.md
3225e114f318eb89806735eb779e82f3605dd5886d6c00d6ebf7fff06702b4ec *realsrc2/portlight--README.md
76a63f60dcc82c4eaf2131d01a34b44f29fdd7e15790e6a421316d7e8d2b3675 *realsrc2/prompt-craft--CHANGELOG.md
de1ddfa0b1423fcd210c5a5c2bbc06a504fc9905a14e11e2d77c9fa72741fa22 *realsrc2/prompt-craft--README.md
d7a4882631472c34b662785697845c8ebd84b151859b7c9da0aef7cde063abe3 *realsrc2/research-os--CHANGELOG.md
aa4db61a3afa63e9c105b564e96994ab31cafd332a2b8b244b806d1ec63a301d *realsrc2/research-os--README.md
6c17d08fcff837dfb960f8c36983e60d1e95bd02b4ad38bfb3eab0bcaae2fced *realsrc2/runforge--CHANGELOG.md
39882f789481a8beca23cab400277d60f4827fb905ad608f35c9ab9515514add *realsrc2/runforge--README.md
d682ddd9e273a94c94eae6af8ed74a4236a1c774a748e47575360362a935974b *realsrc2/runforge--session.rs
580bbfd4bffe9e2a0c0d7bfd1d085872599aefe83bd1365e02410dda3a50f59a *realsrc2/stillpoint--CHANGELOG.md
99d92450a8e842dc6a0f1716abc455f96362e3cbc4da7ad438626e755d62cefa *realsrc2/stillpoint--README.md
b9010219027a9736d658f06dadbfe215720f943934bede4d0e7945e846940603 *realsrc2/stillpoint--bin.ts
798d5b7cd59f03ed2b4143355d85f766d4833c5f10502ab8a1df9ba0d4957ebc *realsrc2/vocal-synth-engine--CHANGELOG.md
268c0baf4eecb0d33a66eb177ca0647de71b7428f34260c2ad5e109538a3333e *realsrc2/vocal-synth-engine--README.md
d75900b9712b8515eef2d0b45b86cd63c067debd65ac1bb7154e86ac17085cf7 *sealedsrc/PINS.txt
98176841806d28b2e0cf8daf26605654d6afd7ea25651f90ad1503cad430071b *sealedsrc/offrig--README.md
```
