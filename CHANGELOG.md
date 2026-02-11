# Community Notes Scoring — Performance Optimization Log

This document tracks our ongoing work to make the [Community Notes](https://github.com/twitter/communitynotes) scoring pipeline more efficient. We forked the original X/Twitter repo and are systematically profiling and optimizing the data loading, preprocessing, and scoring steps — primarily targeting pandas operations and I/O bottlenecks.

---

## 1. Rewrite TSV parser to use PyArrow engine

**Date:** 2026-02-01

The original `tsv_parser` function was a hand-rolled wrapper around `pd.read_csv` that enforced a strict column-count check. Whenever X added new columns to their data exports (which happens periodically), the parser would throw a `ValueError` and the entire pipeline would fail.

**Problem:**
- Brittle validation: hard-coded column count assertion broke on schema changes.
- Read the entire file into a string first, then passed a `StringIO` to pandas — unnecessary memory copy.
- ~80 lines of complex parsing, chunking, and NA-conversion logic.

**Solution:**
- Replaced the entire `tsv_parser` function with a streamlined `tsv_reader_single` that calls `pd.read_csv` directly on the file path (no `StringIO` intermediary).
- For **header files**: uses `usecols=columns` so extra columns in the file are silently ignored — no more breakage on schema changes.
- For **headerless files**: uses `names=columns` as before.
- Switched the CSV engine from the default (C parser) to `engine="pyarrow"` for faster parsing.
- Net result: **~80 lines of brittle code removed**, replaced with ~10 clean lines.

---

## 2. Parallelize TSV directory reads with ThreadPoolExecutor

**Date:** 2026-02-04

The ratings data is sharded across ~20 TSV files in a directory. The original code read them sequentially in a list comprehension, meaning each file waited for the previous one to finish.

**Problem:**
- Sequential I/O on sharded files — total read time was the sum of all individual file read times.

**Solution:**
- Wrapped the per-file reads in a `concurrent.futures.ThreadPoolExecutor` so all TSV files in a directory are read in parallel.
- Since the bottleneck is I/O (not CPU), thread-based parallelism is sufficient and avoids multiprocessing overhead.
- The results are collected and concatenated with `pd.concat` as before.

---

## 3. Add Parquet caching layer for TSV files

**Date:** 2026-02-04

Even with PyArrow and parallel reads, parsing large TSV files from scratch on every run is slow. Columnar formats like Parquet are dramatically faster to read.

**Problem:**
- Every pipeline run re-parsed the same TSV files, even when the underlying data hadn't changed.

**Solution — single-file caching:**
- `tsv_reader_single` now checks for a `.parquet` file alongside the `.tsv` (e.g. `ratings-00001.parquet` next to `ratings-00001.tsv`).
- If the Parquet file exists and its modification time is >= the TSV's, it reads from Parquet directly and skips all CSV parsing.
- On a cache miss, it reads the TSV normally and then writes a `.parquet` cache file (Snappy compression) for next time.

**Solution — directory-level caching:**
- `tsv_reader` (directory mode) checks for a single combined `.parquet` file for the entire directory (e.g. `noteRatings.parquet` next to the `noteRatings/` folder).
- If present and newer than all TSVs in the directory, it reads one file instead of 20+ individual files.
- On a cache miss, it reads and concatenates all TSVs (in parallel), then writes the combined Parquet.

**Helper functions added:**
- `_get_parquet_path(tsv_path)` — derives the `.parquet` path from a `.tsv` path.
- `_read_tsv_raw(path, mapping, columns, header)` — extracted the raw TSV read logic into its own function so it can be called independently of the caching layer.

---

## 4. Optimize `remove_duplicate_ratings`

**Date:** 2026-02-04

**Problem:**
- The original code called `pd.DataFrame(ratings.drop_duplicates())` on **all columns**, creating a full copy of the DataFrame.
- Then it ran a separate `groupby([raterParticipantIdKey, noteIdKey]).head(1)` just to assert that the key pairs were unique — essentially deduplicating a second time to verify the first.
- This was both redundant and slow on millions of rows.

**Solution:**
- Replaced with a single `drop_duplicates(subset=[raterParticipantIdKey, noteIdKey], keep="first")`.
- This deduplicates on just the key columns (much faster than comparing all ~40+ columns) and keeps the first occurrence if rows differ on non-key columns.
- Removed the redundant `groupby` assertion entirely.

---

## 5. Optimize `_filter_misleading_notes` — replace merge with map

**Date:** 2026-02-05 | **Before:** ~2 min 20 sec | **After:** ~1 min 20 sec

This function filters the ratings DataFrame to keep only ratings on notes that are classified as misleading (or were deleted but previously scored). It needs two columns from `noteStatusHistory`: `classification` and `createdAtMillis`.

### 5a. Replace `merge` with `Series.map()`

**Problem:**
- The original code did a full left merge of ratings (~millions of rows) with noteStatusHistory (~hundreds of thousands of rows) using the `safe_merge` wrapper, just to look up two columns by `noteId`.
- `safe_merge` adds type-checking overhead on top of the already expensive merge algorithm.

**Solution:**
- Since `noteStatusHistory` has unique `noteId`s, we set it as the index once and use `Series.map()` to look up each column individually.
- `ratings[classificationKey] = note_ids.map(nsh_index[classificationKey])` — this is an O(n) hash lookup, far cheaper than the full merge.

### 5b. Use local boolean Series instead of DataFrame column writes

**Problem:**
- The original code wrote several temporary boolean columns (`deletedNote`, `notDeletedMisleading`, `deletedButInNSH`) directly into the ratings DataFrame using `ratings[key] = ...`.
- Each write triggers pandas' indexing machinery and potentially copies data.

**Solution:**
- Compute all boolean masks as standalone local Series variables (e.g. `is_deleted = pd.isna(classification)`).
- These never get written into the DataFrame, avoiding the column assignment overhead entirely.
- The final filter uses a single combined mask: `mask = not_deleted_misleading | deleted_but_in_nsh | not_deleted_not_misleading_new_ui`.

### 5c. Replace `np.unique()` with `Series.nunique()`

**Problem:**
- The logging code used `len(np.unique(ratings.loc[mask, noteIdKey]))` — this extracts a numpy array, sorts it, and counts unique values.

**Solution:**
- Replaced with `noteIds[mask].nunique()` — pandas' built-in uses a hash-based approach, avoiding the sort. Cleaner and faster.

### 5d. Simplify final column cleanup

**Problem:**
- The original code dropped temporary columns with an explicit list in `ratings.drop(columns=[...])`.

**Solution:**
- Used `ratings.loc[mask, ratings.columns.difference([temp_cols])]` to select only the desired columns in one step, combining the row filter and column filter.

---

## 6. Optimize `compute_helpful_num`

**Date:** 2026-02-05

A small but clean improvement.

**Problem:**
- Used `ratings.dropna(subset=[helpfulNumKey])` which creates an intermediate subset check.

**Solution:**
- Replaced with `ratings[ratings[helpfulNumKey].notna()]` — more direct boolean indexing, avoids the `dropna` wrapper overhead.

---

## 7. Optimize `compute_affinity_and_coverage` — precompute shared data

**Date:** 2026-02-05 | **Before:** ~3.5 minutes | **After:** ~1.5 minutes

This function in `post_selection_similarity.py` computes rater affinity and writer coverage metrics across multiple time windows (1 minute, 5 minutes, 20 minutes).

### 7a. Precompute the ratings-notes merge

**Problem:**
- The original `_compute_affinity_and_coverage` method was called once per latency window (3 times total).
- Each call independently merged the ratings DataFrame with the notes DataFrame and computed the `latency` column — identical work repeated 3 times.

**Solution:**
- Extracted a new `_merge_ratings_and_notes()` method that merges ratings with notes and computes the latency column **once**.
- The returned merged DataFrame is passed into each per-window call, which only needs to filter by `latency <= threshold`.

### 7b. Precompute writer totals

**Problem:**
- Each per-window call also independently computed `writerTotals` (a `value_counts()` on the note author column) — again, identical across all 3 calls since it depends on notes, not ratings.

**Solution:**
- Extracted a new `_compute_writer_totals()` method that computes per-author note counts **once**.
- Passed into each per-window call alongside the pre-merged DataFrame.

### 7c. Restructured method signatures

**Problem:**
- The original method `_compute_affinity_and_coverage(self, ratings, notes, latencyMins, minDenom)` accepted raw ratings and notes and did everything internally.

**Solution:**
- Renamed to `_compute_affinity_and_coverage_for_latency(self, merged, writerTotals, latencyMins, minDenom)`.
- Now accepts the pre-computed merged DataFrame and writer totals, and only handles the per-window logic: filtering by latency, computing rater totals, computing pair totals, and calculating the affinity/coverage ratios.

### 7d. Select only needed columns early

**Problem:**
- The original code passed the full ratings and notes DataFrames into the per-window method, carrying many unused columns through the merge.

**Solution:**
- Before merging, we select only the 3 columns actually needed from each DataFrame:
  - ratings: `[raterParticipantIdKey, noteIdKey, createdAtMillisKey]`
  - notes: `[noteAuthorParticipantIdKey, noteIdKey, createdAtMillisKey]`
- This makes the merge and all subsequent operations cheaper since the DataFrame is much narrower.

---

## 8. Rewrite `_get_pair_counts_dict` — eliminate hash table bottleneck

**Date:** 2026-02-08 | **Before:** >1 hour / OOM at full scale | **After:** ~5 min

This function counts how many tweets each pair of raters co-rated within a sliding time window. It is the most expensive step in the post-selection similarity pipeline. At full scale (~180M ratings, ~2M note groups), the original implementation produced ~1.5 billion unique rater pairs, consumed ~93GB of RAM in a single Python dict, and was killed by the OOM killer when the hash table tried to resize.

### What the original did

The original used nested `pandas.groupby` loops (tweets → notes) with a pure-Python sliding window and two Python dicts: `pair_counts` (pair → co-rating count) and `pairs_counted_in_tweet` (a set, recreated per tweet for dedup). Every pair comparison involved a hash table lookup into a dict that grew to 1.5B entries and ~93GB — far too large for CPU cache, making each lookup ~100-200ns (main-memory latency).

### What the new version does

The rewrite separates **pair finding** (Numba-compiled, outputs to arrays) from **dedup + counting** (numpy sort + vectorized ops), eliminating hash tables entirely.

**Phase 1 — Preprocessing (numpy, ~30s):**
- Factorize rater/tweet/note IDs to int32 codes.
- Sort by `(note, time)` using `np.lexsort` and compute note-group boundaries vectorially.

**Phase 2 — Windowed pair finding (Numba, ~2 min):**
- Two compiled passes over the sorted data:
  1. **Count pass**: counts how many `(tweet, rater_l, rater_r)` pair events the sliding window will emit — used to pre-allocate output arrays exactly.
  2. **Fill pass**: writes the events into pre-allocated int32 arrays. Same sliding-window logic as the original, but writes to contiguous memory instead of doing hash table lookups.
- The per-note time window (`windowMillis`, default 20 min) is preserved exactly.

**Phase 3 — Dedup + counting (numpy, ~5 min):**
- Packs each `(rater_l, rater_r, tweet)` triple into a single int64 using bit-shifting (dynamic bit allocation based on actual ID ranges). This allows a single `ndarray.sort()` call instead of `np.lexsort` over 3 separate arrays (~3x faster).
- Deduplicates per-tweet pairs by comparing consecutive values in the sorted compound array (one vectorized comparison instead of three).
- Counts co-rated tweets per pair via run-length encoding on the pair key (`compound >> tweet_bits`).
- Pre-filters by `minCoRatingCount` before creating the output dict, reducing ~1.5B pairs to ~millions.

### Pre-filter threshold (`minCoRatingCount`)

The output dict only includes pairs whose count could survive the downstream PMI/minSim filter. The threshold is computed by a new helper, `_min_survivable_co_rating_count`, which lives directly above the filter function it mirrors. It checks both filter conditions (minSim and NPMI) with the most favorable rater totals and takes the lower minimum. With default parameters this evaluates to 8.

### Results (full scale: 180M ratings, 2.15M note groups)

| Stage | Time | RSS |
|-------|------|-----|
| Preprocessing (factorize, sort, groups) | ~90s | 39GB |
| Numba count pass | 4s | 39GB |
| Numba fill pass (2.2B pair events) | 11s | 62GB (peak) |
| Compound key pack + sort | 58s | 54GB |
| Dedup (1.8B unique tweet-pair events) | 11s | 51GB |
| Count + filter (1.53B pairs → 3.66M with count ≥ 8) | 19s | 37GB |
| Dict creation | 6s | 38GB |
| **Total** | **3.3 min** | **62GB peak** |

The pre-filter is critical: 1.53 billion unique pairs collapse to 3.66 million (0.24%) after applying the `minCoRatingCount=8` threshold.  The downstream PMI/minSim filter then runs in 9 seconds on the small dict, vs. the hours it would take on 1.53B entries.

Peak 62GB vs. the original's 93GB (which then OOM'd on resize). No hash table at any point.
