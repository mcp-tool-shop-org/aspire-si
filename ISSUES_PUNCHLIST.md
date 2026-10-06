# Aspire AI — Performance-Focused GitHub Issues Punch List

One issue per bullet. Each includes acceptance criteria inline.

## P0 — Must-fix (stop avoidable extra work)

### Issue: Avoid pretty-printed cache writes on hot path
**Priority:** P0  
**Area:** DialogueManager cache I/O

**Problem:** `_save_to_cache()` uses `json.dump(..., indent=2)` which increases CPU and file size substantially when caching lots of dialogues.

**Work:**
- Add a config flag `cache_pretty: bool = False` (default False).
- Use compact JSON when pretty is False: `separators=(",", ":")`, no indent.

**Acceptance criteria:**
- Default cache files are compact JSON (no indentation).
- With `cache_pretty=True`, behavior matches current formatting.
- Unit test verifies compact output has no leading indentation and parses correctly.

---

### Issue: Add file-locking around cache read/write to prevent corruption under parallel generation
**Priority:** P0  
**Area:** DialogueManager concurrency

**Problem:** `get_dialogues()` can generate concurrently and then write cache files; parallel processes can also access the same cache dir. Without locking, cache can corrupt or partially write.

**Work:**
- Add an `asyncio.Lock` for in-process protection.
- Add optional cross-process lock via lockfile (best-effort).
- Write cache files atomically (write temp + rename).

**Acceptance criteria:**
- Concurrent cache writes do not produce JSON decode errors in `_load_from_cache`.
- Cache writes are atomic (no partially written JSON files).
- Unit test simulates concurrent writes and verifies cache remains readable.

---

### Issue: Add a bounded in-memory cache for repeated prompts in the same run
**Priority:** P0  
**Area:** DialogueManager hot path

**Problem:** Even with disk cache, repeated prompt requests in the same run still do filesystem checks and JSON parsing.

**Work:**
- Add an LRU in-memory cache keyed by prompt+teacher for `get_dialogue`/`get_dialogues`.
- Default size (e.g., 256) configurable.

**Acceptance criteria:**
- Second `get_dialogue(prompt)` in same process returns without filesystem read.
- Unit test monkeypatches `_load_from_cache` and asserts it is not called on second hit when LRU has it.

---

## P1 — High leverage (throughput + batching)

### Issue: Deduplicate prompts in `get_dialogues()` to reduce batch size
**Priority:** P1  
**Area:** batching efficiency

**Problem:** If `prompts` contains duplicates, `generate_batch` may be invoked with redundant items.

**Work:**
- Deduplicate misses before calling `generate_batch`.
- Map results back to original indices.

**Acceptance criteria:**
- `generate_batch` is called with unique prompts only.
- Output ordering matches input ordering.
- Unit test: repeated prompts results in a single batch entry per unique prompt.

---

### Issue: Adaptive concurrency based on teacher rate limits
**Priority:** P1  
**Area:** async performance

**Problem:** Fixed `max_concurrent` may overload external teacher APIs or underutilize local teachers.

**Work:**
- Add a simple adaptive limiter:
  - for remote teachers, cap concurrency lower by default
  - for local teachers, allow higher
- Allow per-teacher override.

**Acceptance criteria:**
- Concurrency defaults differ by teacher type (remote vs local).
- Unit test asserts correct max_concurrent passed to `generate_batch` for each teacher type.

---

## P2 — Maintainability & regression-proofing

### Issue: Add stable perf regression tests (call-count based, not timing)
**Priority:** P2  
**Area:** tests

**Work:**
- Maintain tests that assert:
  - cache hits avoid generator calls
  - batch generation is called only for misses
  - cache key differs across teachers (prevents accidental cross-teacher reuse)
- Add `-m perf` marker for optional extended suite (if desired).

**Acceptance criteria:**
- Perf tests pass reliably on CI with no wall-clock thresholds.
- Tests run without requiring GPU or external APIs.

