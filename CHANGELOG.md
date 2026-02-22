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

---

## 9. Rewrite `_get_pair_counts` in quasi-clique detection

**Date:** 2026-02-08 | **Before:** ~40s (10% sample) | **After:** ~3s (10% sample)

The `_get_pair_counts` method in `QuasiCliqueDetection` counts how many tweets each rater pair rates in the same way (same note, same `helpfulNum`), deduped per tweet. Structurally identical to the PSS pair counting but simpler: no time window, and the data is pre-filtered to recent misleading notes (13 weeks), making it much smaller.

### What the original did

Used `pandas.groupby(...).agg(list)` to build nested Python lists of raters, then a triple-nested Python loop over tweets → note groups → rater pairs, with a per-tweet `set()` for dedup and a global `dict()` for counting. At 10% sample: ~97K tweets iterated one-by-one with per-1000 progress logging.

### What the new version does

Same Numba + array + compound-key approach as PSS entry #8:

1. Factorize IDs, sort by `(noteId, helpfulNum)`, compute group boundaries.
2. Compute total pair events exactly from group sizes — `sum(k*(k-1)/2)` — no Numba count pass needed (no time window means every pair in a group counts).
3. Single Numba pass fills pre-allocated int32 arrays with `(tweet, rater_l, rater_r)` events.
4. Pack into int64 compound key, in-place sort, vectorized dedup, run-length count.
5. Filter by `minAlignedRatings` (default 5) and map codes back to original IDs.

### Results (10% sample: 1.4M filtered ratings, 202K groups)

| | Original | New |
|---|---|---|
| Pair counting | ~21s (Python loop over 97K tweets) | ~2s (Numba fill + numpy sort on 29M events) |
| Total `_get_pair_counts` | ~27s | ~4s |
| Total quasi-clique step | 40s | 16s |
| Output pair counts | 4,643 | 4,643 (identical) |

---

## 10. Rewrite `_build_clusters` / `_grow_clique` — eliminate pandas from inner loop

**Date:** 2026-02-12 | **Before:** ~308 min (21% sample) | **After:** ~5 min (21% sample)

The `_build_clusters` method repeatedly calls `_grow_clique` to greedily grow quasi-cliques one rater at a time. Each `_grow_clique` call iterates up to `maxCliqueSize` (2000) times, and each iteration performed multiple full DataFrame merges, `value_counts()`, `isin()` filters, and `drop_duplicates()` calls against the entire `raterPairRatings` DataFrame. At scale this meant hundreds of full pandas passes over millions of rows per clique.

### What the original did

Each iteration of the inner loop:
1. Filtered `raterPairRatings` with `.isin(includedRaters)` — full DataFrame scan per iteration.
2. Computed `value_counts()` on `(tweetId, noteId, helpfulNum)` to find group actions — sort-based, O(N).
3. Merged ALL `raterPairRatings` with qualifying actions to find aligned non-included raters — full inner join, O(N).
4. Ran `drop_duplicates()` + `value_counts()` to count unique tweets per candidate — another O(N) pass.
5. Repeated steps 1–4 for the trial-add check (with candidate included).
6. Performed a second merge + `drop_duplicates()` + `value_counts()` to check per-rater inclusion thresholds.

Total: **~6 full DataFrame passes per iteration**, hundreds of iterations per clique, multiple cliques.

### What the new version does

A new `_prepare_grow_clique_arrays` method is called **once** before the clique-building loop. It converts the entire `raterPairRatings` DataFrame into numpy arrays and CSR (Compressed Sparse Row) inverted indices:

**One-time setup (`_prepare_grow_clique_arrays`):**
1. Factorize all IDs (rater, tweet, note, helpfulNum) to contiguous int32 codes.
2. Bit-pack `(tweet, note, helpfulNum)` triples into compound int64 action codes, then re-factorize to contiguous int32 action codes.
3. Build **rater → actions** CSR index (`rat_act_indptr` + `rat_act_data`) — O(1) lookup of any rater's actions.
4. Build **action → raters** CSR index (`act_rat_indptr` + `act_rat_data`) — O(1) lookup of which raters share an action.
5. Build `tweet_of_action` mapping and `rater_id_to_code` reverse lookup dict.

**Per-iteration of `_grow_clique` (all numpy, no pandas):**
1. **Qualifying actions**: `action_count >= threshold` — single vectorized comparison on a pre-allocated int32 array. Action counts are maintained **incrementally**: when a rater is added, only their actions are incremented (`action_count[rater_actions] += 1`), O(k) where k = that rater's number of actions.
2. **Candidate selection**: gather all `(rater, tweet)` pairs from qualifying actions via the action→raters CSR index, filter to non-included raters, deduplicate with bit-packed compound key + `np.unique`, count unique tweets per rater with `np.bincount`, pick `argmax`.
3. **Trial-add**: temporarily increment the candidate's action counts, recompute qualifying actions with updated thresholds, check `satisfiedTweets` and per-rater inclusion thresholds via CSR lookups. Rollback on rejection (`action_count[cand_actions] -= 1`).

### Per-iteration cost comparison

| Operation | Original | New |
|---|---|---|
| Find qualifying actions | `value_counts()` on full DF — O(N) | `action_count >= threshold` — vectorized O(n_actions) |
| Find best candidate | `merge()` + `drop_duplicates()` + `value_counts()` — O(N) | CSR gather + `np.unique` + `np.bincount` — O(qualifying pairs) |
| Trial-add threshold check | second `merge()` + `value_counts()` — O(N) | per-rater boolean index on CSR arrays — O(n_included × k) |
| Action count update | full recompute from scratch — O(N) | `action_count[cand_actions] += 1` — O(k) |

Where N = total rows in `raterPairRatings` (millions) and k = one rater's actions (hundreds).

### Results (21% sample)

| | Original | New |
|---|---|---|
| Compute Quasi-Cliques | 308 min | 5 min |
| **Speedup** | | **~63x** |

### Note on tie-breaking

Candidate selection tie-breaking differs between the two implementations: the original uses pandas `value_counts()` hash-table ordering, while the new version uses `np.argmax` (lowest rater code among ties). Since the algorithm is a greedy heuristic, different tie-breaking produces slightly different (but equally valid) clique assignments — ~3% difference in total rater assignments at 21% sample. All cliques satisfy the same density constraints.

---

## 12. Optimize `apply_post_selection_similarity` — replace DataFrame merges with dict lookups

**Date:** 2026-02-14 | **Before:** ~8 min (full scale) | **After:** ~2 min (full scale)

This function filters ratings from PSS-flagged raters: dropping ratings where rater and note author are in the same clique, and deduplicating so only the earliest rating per (noteId, postSelectionValue) pair survives. At full scale (~197M ratings), it was the single slowest step remaining in the prescoring pipeline.

### What the original did

Three chained `DataFrame.merge()` calls on the full ratings DataFrame:
1. Left-join ratings with PSS values on `raterParticipantId` — copies all ~40 columns for every row.
2. Left-join with notes on `noteId` — copies everything again to add `noteAuthorParticipantId`.
3. Left-join with PSS values again on `noteAuthorParticipantId` — third full copy.

Then split the merged DataFrame into two subsets (has PSS / no PSS), sort + dedup one subset, and `pd.concat()` them back together. Two `drop_duplicates()` calls for diagnostic logging added ~2 minutes of pure overhead.

### What the new version does

1. **Dict lookups instead of merges**: builds two small dicts (`rater_to_pss` mapping rater → clique ID, `note_to_author` mapping noteId → author) and uses `Series.map()` for O(n) hash probes with no DataFrame copying.
2. **Boolean mask instead of split/concat**: a single `drop_mask` Series identifies rows to remove (same-clique + PSS-dedup duplicates). One `ratings[~drop_mask]` operation filters in-place.
3. **Dedup on 3-column slice**: the sort + `duplicated()` for PSS dedup operates on just `[noteId, createdAtMillis, pss_value]` extracted from the PSS-flagged subset (typically <1% of rows), not the full wide DataFrame.
4. **Removed logging `drop_duplicates()`**: two calls to `ratings[[noteId, raterParticipantId]].drop_duplicates()` that existed purely for log messages (~55s each) were removed.

### Dev cache checkpoint moved

The dev cache checkpoint was moved from `pre_apply_pss` (before PSS filtering) to `pre_get_scorers` (after PSS filtering, right before `_get_scorers()`). The cached data now includes the post-PSS ratings DataFrame, so on cache hit the pipeline skips rater clustering, topic model, **and** PSS filtering.

---

## 11. Rewrite `_make_seed_labels` — eliminate regex with plain string search

**Date:** 2026-02-12 | **Before:** ~8 min (full scale) | **After:** ~45s (full scale)

The `_make_seed_labels` method in `TopicModel` assigns topic labels to notes by matching seed terms against note text. At full scale (~500K+ texts), the original implementation took ~8 minutes.

### What the original did

A single combined regex with named groups (one per topic) was compiled from all ~20 seed terms. For every text, `finditer()` was called to find all matches, then a Python `set()` collected which named groups fired, with `Topics[grp].value` lookups for each match. This meant ~500K Python iterations, each creating regex Match objects, dicts (`groupdict()`), sets, and doing enum lookups.

### What the new version does

Eliminates regex entirely by converting all seed patterns to plain string searches using Python's `in` operator (C-level Boyer-Moore):

1. **Whitespace normalization**: `str.translate()` replaces all whitespace characters with spaces in a single C-level pass per text.
2. **Prepend-space trick**: a space is prepended to each text so that searching for `" term"` handles both `\s` (whitespace before term) and `^` (term at start of string) boundary conditions in one plain check.
3. **Pattern conversion**:
   - Simple patterns like `"ukrain"` → search for `" ukrain"`
   - Patterns with `\s` like `"\shamas\s"` → replace `\s` with space → search for `"  hamas "` (preserves original boundary semantics)
   - URL patterns with `\.` like `"help\.x\.com"` → search for `"help.x.com"` (no boundary prefix)
4. **Vectorized matching**: `pd.Series.str.contains(term, regex=False)` runs C-level substring search across all texts for each term, with results combined via numpy boolean OR per topic.
5. **Removed `_compile_regex`**: the compiled regex and its `_compile_regex()` builder method are no longer needed and were removed.

### 11b. Cache `custom_tokenizer` results and pre-compute its components

The `custom_tokenizer` method was recreating a `CountVectorizer` preprocessor and recompiling a regex pattern on every single call (~500K+ calls). Worse, the same texts are tokenized twice: once in `_get_stop_words` (to build the vocabulary) and again in the pipeline's `CountVectorizer.fit()` (to build the document-term matrix).

- **Pre-computed components**: moved the preprocessor and compiled regex pattern into `__init__`, eliminating ~500K redundant object instantiations.
- **Result cache** (`_tokenizer_cache`): the first tokenization pass (`_get_stop_words`) populates a dict cache. The second pass (`pipe.fit`) is a dict lookup per text instead of regex matching. In the bootstrapped case, all iterations after the first are fully cached as well.

### Results (full scale)

| | Original | New |
|---|---|---|
| `_make_seed_labels` | ~8 min | ~45s |
| `custom_tokenizer` (2nd pass) | ~3 min | ~0s (cached) |
| **Total speedup** | | **~10x + ~1 min saved on training** |

---

## 13. Convert participant IDs from strings to integers at data load time

**Date:** 2026-02-17

In dev, participant IDs contain letters (e.g. `"abc123"`), so the `Int64Dtype()` conversion in `run_prescoring` always fails silently, leaving IDs as Python string objects throughout the entire pipeline.

**Problem:**
- String/object columns in pandas store each value as a separate heap-allocated Python object (~50-80 bytes each) plus an 8-byte pointer, vs. 8 bytes per value for `Int64`.
- Every merge, join, groupby, and `isin` on participant ID columns pays for string hashing (O(n) per string) and character-by-character comparison, plus poor CPU cache locality from pointer chasing.
- The existing try/except in `run_prescoring` caught the `ValueError` and logged a message, but left the pipeline running with string IDs for all of prescoring and final scoring.

**Solution:**
- Added a `pd.factorize()`-based conversion in `runner.py` `_run_scorer`, immediately after data loading and sampling — the earliest possible point.
- Collects all unique participant IDs from `ratings[raterParticipantIdKey]`, `statusHistory[noteAuthorParticipantIdKey]`, `userEnrollment[participantIdKey]`, and `notes[noteAuthorParticipantIdKey]` into one Series.
- `pd.factorize()` produces a single consistent string-to-int mapping; `.map()` replaces all four columns in-place.
- No reversal needed — participant IDs are opaque join keys, so the pipeline and outputs work identically with integer IDs.
- The dev cache stores the already-converted DataFrames, so subsequent cached runs also benefit.
- The existing `Int64Dtype()` conversion in `run_prescoring` now succeeds instead of falling through to the except branch.
- Fixed the post-prescoring ID restoration in `run_prescoring`: the original code used `.astype(str)` to "restore" IDs after prescoring's temporary `Int64` conversion. This was designed for the old string-ID regime and left `ratings`, `noteStatusHistory`, and `userEnrollment` as **object/str** while `notes` remained **int64** (since it was never part of the conversion). Changed the restoration to `.astype(np.int64)` so all DataFrames stay type-consistent with the factorized IDs. Also saves ~8 GB of memory (int64 = 8 bytes/value vs. str objects = ~50+ bytes/value for ~190M ratings).

---

## 14. Enable GPU/MPS acceleration for reputation matrix factorization models

**Date:** 2026-02-19

The reputation MF subsystem (diligence model, helpfulness model) hardcoded `device=torch.device("cpu")` in every function signature, while the main `BiasedMatrixFactorization` model in `matrix_factorization/model.py` auto-detected CUDA/MPS at init time. This meant the reputation models always ran on CPU even when a GPU was available.

**Problem:**
- All functions in `diligence_model.py`, `helpfulness_model.py`, `reputation_matrix_factorization.py`, and `dataset.py` defaulted to `torch.device("cpu")`.
- Callers in `mf_base_scorer.py` never passed a `device` argument, so the default always took effect.
- `_setup_model` in `reputation_matrix_factorization.py` did not pass `device` through to `ReputationMFModel` at all — a latent bug that would have caused a device mismatch if anyone had tried to pass a non-CPU device.

**Solution:**
- Added a shared `detect_device()` function in `dataset.py` that mirrors the detection logic from `BiasedMatrixFactorization`: prefers `cuda:0` → `mps` → `cpu`.
- Changed all `device=torch.device("cpu")` defaults to `device=None` across 4 files (10 function signatures total), with each function resolving `None` to `detect_device()` at call time.
- Fixed `_setup_model` to accept and forward the `device` parameter to `ReputationMFModel`.
- Callers that don't pass a device get automatic GPU detection; callers that pass an explicit device still work as before.

**Files changed:**
- `reputation_matrix_factorization/dataset.py` — added `detect_device()`, updated `build_dataset`
- `reputation_matrix_factorization/reputation_matrix_factorization.py` — updated `ReputationMFModel.__init__`, `_setup_model`, `train_model_prescoring`, `train_model_final`
- `reputation_matrix_factorization/diligence_model.py` — updated `_setup_dataset_and_hparams`, `fit_low_diligence_model_final`, `fit_low_diligence_model_prescoring`
- `reputation_matrix_factorization/helpfulness_model.py` — updated `_setup_dataset_and_hparams`, `get_helpfulness_reputation_results_final`, `get_helpfulness_reputation_results_prescoring`

---

## 15. Aggressive memory cleanup to prevent OOM kills during scoring

**Date:** 2026-02-21

The pipeline was being killed by the OOM killer during prescoring — specifically during the MFExpansionScorer's `compute_tag_thresholds_for_percentile` step. At that point, ~55-60 GB of DataFrames were alive simultaneously across multiple layers of the call stack: two full copies of the ratings data (~19 GB each), plus scorer intermediates that were never freed.

**Problem — large DataFrames kept alive unnecessarily:**
- `ratingsForTraining` (~15-19 GB) inside each scorer's `_prescore_notes_and_users` was only deleted at the very end of the method, long after its last real use (creating `finalRoundRatings`).
- `scoredNotes` (first computation), `helpfulnessScoresPreHarassmentFilter`, and `harassmentAbuseNoteParams` were never explicitly freed despite being superseded by later computations.
- `compute_tag_thresholds_for_percentile` created several large intermediate DataFrames (`tagAggregates`, merged `scoredNotes`, `crhNotes`, `crhStats`) and returned without deleting any of them.
- The `cached` dict from `dev_cache.load("pre_get_scorers")` held redundant references to all its values alongside the extracted local variables.
- After prescoring, `cachedRatings`, `prescoringNotesInput`, `prescoringRatingsInput`, and `postSelectionSimilarityValues` remained alive in `run_scoring`'s scope despite never being used again.
- No `gc.collect()` between serial scorer runs — each scorer's garbage accumulated until Python's cyclic GC threshold was eventually hit.

**Solution — `mf_base_scorer.py`:**
- Delete `ratingsForTraining` immediately after creating `finalRoundRatings` (saves ~15 GB before the expensive tag threshold + diligence steps).
- Delete `scoredNotes` (first version) and `helpfulnessScoresPreHarassmentFilter` after their last use in the post-harassment helpfulness score computation.
- Delete `harassmentAbuseNoteParams` after its merge into `noteParams` in the diligence block.
- Delete `noteParamsUnfiltered` and `raterParamsUnfiltered` at end of method.
- In `compute_tag_thresholds_for_percentile`: delete `tagAggregates`, `scoredNotes`, `crhNotes`, `crhStats` and call `gc.collect()` before returning.
- In `_score_notes_and_users` (final scoring path): delete `ratingsForTraining` after creating `finalRoundRatings`.
- All deletions gated on `not self._saveIntermediateState` where the variable is conditionally saved to `self`.

**Solution — `run_scoring.py`:**
- `del cached` immediately after extracting values from the dev cache dict.
- `del cachedRatings, cachedNoteTopics, cachedNoteTopicClassifierPipe` inside `run_prescoring`'s cache-hit branch after assigning to local variables.
- `del prescoringNotesInput, prescoringRatingsInput, postSelectionSimilarityValues, cachedRatings, cachedNoteTopics, cachedNoteTopicClassifierPipe` + `gc.collect()` in `run_scoring` after prescoring returns.
- Converted the serial scorer list comprehension in `_run_scorers` to a `for` loop with `gc.collect()` after each scorer completes.

---

## 16. Fix participant-ID dtype mismatches that crash merges/concats after factorization

**Date:** 2026-02-22

After converting participant IDs from strings to dense `int64` via `pd.factorize` (entry 13), several code paths still created DataFrames where ID columns defaulted to `object` or `float64` — particularly when results were empty (0 rows). This caused `ValueError: You are trying to merge on object and float64 columns` crashes during `run_rater_clustering` and would surface in other merge/concat operations as well.

**Root cause:** `pd.DataFrame(columns=[...])` creates all columns as `object` dtype, and `pd.DataFrame({"key": []})` defaults to `float64`. When these empty DataFrames are merged or concatenated with non-empty DataFrames that have `int64` ID columns, pandas raises a type-mismatch error.

**Fixes:**

- **`scorer.py`** — Added `_empty_df_typed()` helper that creates empty DataFrames with correct `int64` dtype for known ID columns (`raterParticipantId`, `noteAuthorParticipantId`, `participantId`, `noteId`). Updated `prescore()` and `_return_empty_final_scores()` to use it.
- **`post_selection_similarity.py`** — `get_post_selection_similarity_values()`: when `cliquesDfList` is empty, return a DataFrame with explicit `int64` / `Int64` dtypes instead of the default `object`. Also cast `raterParticipantId` to `int64` in the non-empty path.
- **`quasi_clique_detection.py`** — `get_quasi_cliques()`: wrap `raterIds` in `pd.array(..., dtype=np.int64)` so the column is `int64` even when empty.
- **`gaussian_scorer.py`** — Two early-return paths that returned bare `pd.DataFrame()` (no columns at all) now raise `EmptyRatingException`, which the caller in `scorer.py` already catches and handles with properly-typed empty DataFrames. Also fixes a latent `NameError` in the "not enough raters" branch where `quantile_range` was never set.
- **`mf_base_scorer.py`** — Same pattern: the `len(finalRoundRatings) == 0` early return in `_score_notes_and_users` now raises `EmptyRatingException` instead of returning bare `pd.DataFrame()`.
- **`matrix_factorization/pseudo_raters.py`** — Removed the `str()` conversion of extreme-rater IDs (line 223, a relic of the string-ID era). Updated the dtype-matching logic to use `astype(existingIdDtype)` instead of only handling `pd.Int64Dtype`, so it works with both `Int64` (prescoring) and `int64` (final scoring).
- **`mf_base_scorer.py`** — Removed overly-strict assertions that hardcoded which specific scorers were "allowed" to have zero ratings after filtering (`MFTopicScorer_MessiRonaldo`, `MFGroupScorer_33`). Any group/topic scorer can legitimately end up with zero ratings on small subsets; the `EmptyRatingException` mechanism already handles this correctly.
- **`pflip_plus_model.py`** — `_compute_scoring_cutoff()`: the `.astype(pd.Int64Dtype())` calls converted the entire DataFrame (including `noteIdKey`) from `int64` to `Int64`, creating a type mismatch when merging with `scoringCutoff`. Fixed by applying `Int64` only to the timestamp value columns. Also added `as_index=False` to the `groupby()` calls and changed the final merge to `how="left"` (with explicit `on=c.noteIdKey`) so notes without ratings get NaN instead of being dropped. Cast `notes[createdAtMillisKey]` to `Int64` alongside the existing `noteStatusHistory[createdAtMillisKey]` cast to keep merge keys consistent.
- **`pflip_plus_model.py`** — Replaced 8 additional `.astype(pd.Int64Dtype())` calls across `_get_quick_rating_stats`, `_get_burst_rating_stats`, `_get_recent_rating_stats`, `_get_viewpoint_rating_stats`, and `_get_peer_note_stats` with column-specific `int64` casts that skip `noteIdKey`. Same root cause: blanket `Int64` cast was converting `noteIdKey` and breaking downstream merges. Also switched all `groupby(c.noteIdKey)` calls in these functions to use `as_index=False` (removing redundant `.reset_index()`), except for chains that call `.abs()` between groupby and reset (where `noteIdKey` must stay in the index to avoid abs modifying it). Fixed empty DataFrame initialization in `predict()` to use explicit `dtype=np.int64` for `noteIdKey`.
- **`scoring_rules.py`** — Fixed two empty DataFrame returns in `PopulationSampledIntercept.score_notes()` where `pd.DataFrame({c.noteIdKey: [], statusColumn: []})` defaulted to `float64` / `object`. Added explicit `dtype=np.int64` for the ID column.
- **`note_ratings.py`** — Fixed two no-op `astype()` calls in `get_note_counts_by_rater_sign()` and `get_population_sampled_counts_by_rater_sign()` where the result was not assigned back to the column, leaving a potential type mismatch between `raterModelOutput` and `ratings` on the `raterParticipantId` merge key.
- **`run_scoring.py`** — Added `quasiCliqueValueKey` to the `unsafeAllowed` set in the `prescoringRaterModelOutput` concat, matching the existing allowance for `postSelectionValueKey`.
- **`mf_base_scorer.py`** — `score_final()` overrides the base class but did not catch `EmptyRatingException` from any of its four `_score_notes_and_users()` calls (no-high-vol, no-correlated, population-sampled, and main scoring). When a scorer like `MFGroupScorer_12` has very few ratings and the population-sampled subset filters to zero after helpfulness scoring, the uncaught exception crashed the run. Wrapped all four calls in `try/except EmptyRatingException`, using empty DataFrames for subsidiary calls and `_return_empty_final_scores()` for the main call.
- **`gaussian_scorer.py`** — Same fix: `score_final()` has the identical four unprotected `_score_notes_and_users()` calls. Added `try/except EmptyRatingException` around all four.

