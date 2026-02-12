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

# Second run — cached stages are skipped, jumps straight to Quasi-Cliques:
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

### 2. `pss_complete.pkl` — Full Post Selection Similarity Output

**Saved in:** `run_scoring.py` → `run_rater_clustering()`,
right *before* `Compute Quasi-Cliques` is called.

**Contains:**

| Key                            | Type             | Description                                                      |
|--------------------------------|------------------|------------------------------------------------------------------|
| `postSelectionSimilarityValues`| `pd.DataFrame`   | Final PSS output (rater → clique mapping)                        |

**What it skips on cache hit:**
- Helpful ratings filtering
- `compute_affinity_and_coverage()` (~3.5 minutes)
- `get_suspect_pairs()`
- `_preprocess_ratings()` (~2.5 minutes)
- `_get_pair_counts_dict()`
- PMI/MinSim computation
- `aggregate_into_cliques()`
- `get_post_selection_similarity_values()`

**What still runs (always):**
- Quasi-Clique detection
- Everything downstream (prescoring, final scoring, contributor scoring)

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
       ├─ run_rater_clustering()
       │    │
       │    ├─ [CACHE: pss_complete] ── cache hit?  → load from cache, skip PSS
       │    │                             cache miss? → compute PSS & save
       │    │
       │    ├─ PostSelectionSimilarity.__init__()    ← skipped on cache hit
       │    └─ get_post_selection_similarity_values() ← skipped on cache hit
       │
       ├─ run_prescoring()                          ← always runs
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
2. **`run_rater_clustering()`:** Wrapped the entire Post Selection Similarity
   computation in a cache check/save. On cache hit, skips straight to
   Quasi-Clique detection.

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

# Delete just the PSS cache (re-run full PSS but keep data loading cache):
rm .cache/pss_complete.pkl

# Delete just the data loading cache (re-parse TSVs but keep PSS cache):
rm .cache/runner_data.pkl
```

---

## Cache File Sizes

Expect the cache files to be large — they contain full DataFrames:

- `runner_data.pkl`: Typically **several GB** (contains all input DataFrames).
- `pss_complete.pkl`: Typically **small** (just the rater-to-clique mapping DataFrame).

Make sure the cache directory has sufficient disk space, and add it to
`.gitignore` if it's inside the repo.

---

## Disabling

Simply omit the `--cache-dir` flag. All cache logic becomes a no-op — zero
overhead, no behavior change. The cache module checks `is_enabled()` at the top
of every `save()`/`load()` call and returns immediately when disabled.
