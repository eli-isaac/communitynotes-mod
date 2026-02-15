# Dev Cache — Skipping Expensive Pipeline Stages During Development

## Problem

A full scoring run takes ~10 minutes. Most of that time is spent on stages
*before* the code you're actively working on (data loading, affinity/coverage
computation, ratings preprocessing, etc.). Re-running the entire pipeline on
every code change is painful.

## Solution

A lightweight, opt-in pickle cache controlled by a single CLI flag: `--cache-dir`.
When enabled, the pipeline saves intermediate results to disk after expensive
stages. On subsequent runs, those stages are skipped entirely and the cached
data is loaded instead — typically cutting startup time from ~10 minutes to
under 30 seconds.

---

## Quick Start

```bash
# First run — full pipeline, caches are written:
python main.py \
  --notes data/notes-00000.tsv \
  --ratings data/ratings-00000.tsv \
  --status data/noteStatusHistory-00000.tsv \
  --enrollment data/userEnrollment-00000.tsv \
  --outdir data \
  --cache-dir .cache

# Second run — cached stages are skipped, jumps straight to _get_scorers:
python main.py \
  --notes data/notes-00000.tsv \
  --ratings data/ratings-00000.tsv \
  --status data/noteStatusHistory-00000.tsv \
  --enrollment data/userEnrollment-00000.tsv \
  --outdir data \
  --cache-dir .cache
```

To force a full re-run, delete the cache directory:

```bash
rm -rf .cache
```

---

## What Gets Cached

There are **two cache files**, each covering a different section of the pipeline.

### 1. `runner_data.pkl` — Input Data Loading

**Saved in:** `runner.py` → `_run_scorer()`, after data loading and rating sampling.

**Contains:**

| Key                        | Type                 | Description                                           |
|----------------------------|----------------------|-------------------------------------------------------|
| `notes`                    | `pd.DataFrame`       | The loaded notes dataset                              |
| `ratings`                  | `pd.DataFrame`       | The loaded (and optionally sampled) ratings dataset   |
| `statusHistory`            | `pd.DataFrame`       | The loaded note status history dataset                |
| `userEnrollment`           | `pd.DataFrame`       | The loaded user enrollment dataset                    |
| `previousScoredNotes`      | `pd.DataFrame\|None` | Previously scored notes (if `--previous-scored-notes`) |
| `previousAuxiliaryNoteInfo`| `pd.DataFrame\|None` | Previous aux note info (if `--previous-aux-note-info`) |

**What it skips on cache hit:**
- TSV file parsing (`LocalDataLoader.get_data()`)
- Previous scored notes loading
- Rating sampling (`--sample-ratings`)

### 2. `pre_get_scorers.pkl` — Right Before `_get_scorers`

**Saved in:** `run_scoring.py` → `run_prescoring()`,
right *before* `_get_scorers()` is called (after PSS filtering).

**Contains:**

| Key                            | Type             | Description                                                      |
|--------------------------------|------------------|------------------------------------------------------------------|
| `postSelectionSimilarityValues`| `pd.DataFrame`   | Final PSS output (rater → clique mapping)                        |
| `noteTopics`                   | `pd.DataFrame`   | Note topic assignments from topic model                          |
| `noteTopicClassifierPipe`      | sklearn Pipeline | Trained topic classifier pipeline                                |
| `ratings`                      | `pd.DataFrame`   | Ratings after `apply_post_selection_similarity()` filtering      |

**What it skips on cache hit:**
- `run_rater_clustering()` — PSS computation, quasi-clique detection
- Note topic assignment (topic model training + `get_note_topics()`)
- `apply_post_selection_similarity()` — PSS-based rating filtering

**What still runs (always):**
- `_get_scorers()` and everything downstream

---

## Pipeline Flow With Cache

```
_run_scorer()
  │
  ├─ [CACHE: runner_data] ─── cache hit?  → load from .cache/runner_data.pkl
  │                            cache miss? → load TSVs, sample, save cache
  │
  └─ run_scoring()
       │
       ├─ filter_input_data_for_testing()          ← always runs (fast)
       │
       ├─ [CACHE: pre_get_scorers] ─ cache hit?  → load, skip clustering + topics + PSS
       │                              cache miss? → run all three, save
       │
       ├─ run_rater_clustering()                    ← skipped on cache hit
       │
       ├─ run_prescoring()
       │    ├─ Note topic assignment                ← skipped on cache hit
       │    ├─ apply_post_selection_similarity()    ← skipped on cache hit
       │    ├─ _get_scorers()                       ← always runs (resumes here)
       │    └─ ... rest of prescoring
       │
       ├─ run_final_note_scoring()                  ← always runs
       └─ run_contributor_scoring()                 ← always runs
```

---

## Files Modified

### New file: `scoring/src/scoring/dev_cache.py`

Self-contained cache utility module. Provides:

- `configure(cache_dir)` — Enable caching, create directory.
- `is_enabled()` — Check if caching is active.
- `save(name, data)` — Pickle data to `<cache_dir>/<name>.pkl`.
- `load(name)` — Load from cache. Returns `None` on miss or if disabled.
- `clear(name=None)` — Delete one or all cache files.

All functions are no-ops when caching is disabled (i.e., `--cache-dir` not passed).

### Modified: `scoring/src/scoring/runner.py`

1. **Import:** Added `from . import dev_cache`.
2. **CLI flag:** Added `--cache-dir` argument to `parse_args()`.
3. **`main()`:** Calls `dev_cache.configure(args.cache_dir)` before `_run_scorer`.
4. **`_run_scorer()`:** Wrapped the data-loading block in a cache check/save.

### Modified: `scoring/src/scoring/run_scoring.py`

1. **Import:** Added `from . import dev_cache`.
2. **`run_scoring()` / `run_prescoring()`:** Cache checkpoint is right before
   `_get_scorers()`. On cache hit, skips `run_rater_clustering()`, the topic model,
   and `apply_post_selection_similarity()`; loads cached data (including post-PSS
   ratings) and resumes from `_get_scorers()` and everything downstream.

---

## Cache Invalidation

Caches are **not** automatically invalidated. You must manually delete them when:

- **Input data changes** — e.g., you download a new notes/ratings dataset.
- **Upstream code changes** — e.g., you modify `_preprocess_ratings()` or
  `compute_affinity_and_coverage()`.
- **CLI args change** — e.g., different `--sample-ratings` value.

```bash
# Delete everything:
rm -rf .cache

# Delete just the pre-get-scorers cache (re-run rater clustering + topic model + PSS but keep data loading cache):
rm .cache/pre_get_scorers.pkl

# Delete just the data loading cache (re-parse TSVs but keep PSS cache):
rm .cache/runner_data.pkl
```

---

## Cache File Sizes

Expect the cache files to be large — they contain full DataFrames:

- `runner_data.pkl`: Typically **several GB** (contains all input DataFrames).
- `pre_get_scorers.pkl`: Typically **several GB** (post-PSS ratings, rater-to-clique mapping, note topics, topic classifier).

Make sure the cache directory has sufficient disk space, and add it to
`.gitignore` if it's inside the repo.

---

## Disabling

Simply omit the `--cache-dir` flag. All cache logic becomes a no-op — zero
overhead, no behavior change. The cache module checks `is_enabled()` at the top
of every `save()`/`load()` call and returns immediately when disabled.
