# Community Notes Scoring Pipeline

A high-level conceptual overview of how the scoring pipeline works, from raw input data to final note statuses and contributor scores.

## Input Data

The pipeline starts with four input datasets:

- **Notes** — one row per note. Contains note ID, the tweet it's attached to, the note text, the author, creation time, and classification (whether the author claims the tweet is misleading).
- **Ratings** — one row per rating. Contains which rater rated which note, whether they found it helpful/not-helpful, and which specific tags they selected (e.g., "Incorrect", "MissingKeyPoints", "SpamHarassmentOrAbuse").
- **Note Status History** — one row per note. Tracks the history of each note's status (CRH, CRNH, NMR) and when statuses changed.  Used for inertia (sticky scoring) and flip detection.
- **User Enrollment** — one row per contributor. Contains enrollment state and modeling group assignment.

## Pipeline Stages

### Stage 0: Data Loading and Preprocessing (`runner.py`)

1. Load all four TSV files from disk.
2. **ID factorization** — convert string participant IDs to integers (saves ~50% memory on the ratings table).
3. **Drop unused columns** — remove rating columns never referenced during scoring (`ratedOnTweetId`, `version`, `agree`, `disagree`) to save memory.
4. Pass everything to `run_scoring`.

### Stage 1: Rater Clustering (`run_rater_clustering`)

Before any scoring happens, the system identifies potentially manipulative raters:

- **Post-Selection Similarity (PSS)** — detects raters who systematically rate the same notes in suspiciously similar patterns.  Produces a per-rater similarity score.
- **Quasi-Clique Detection** — identifies tightly connected groups of raters who may be coordinating.

These values are used later to filter out ratings from manipulative raters.

### Stage 2: Prescoring (`run_prescoring`)

Prescoring is the first of two scoring phases.  Its purpose is to solve a chicken-and-egg problem: you need to know which raters are trustworthy to score notes accurately, but you need note scores to determine which raters are trustworthy.  Prescoring breaks this cycle through iteration.

Prescoring runs on **all** notes and ratings (no filtering by recency).

#### 2a. Topic Assignment

A text classifier is trained on note text to assign each note to a topic (e.g., politics, science, health).  Topic scorers will later score notes within each topic independently.

#### 2b. Post-Selection Similarity Filtering

Ratings from raters with high PSS/quasi-clique scores are removed.

#### 2c. Scorer Prescore Calls

Multiple independent scorers each run their `prescore()` method.  The scorers are:

| Scorer | Purpose |
|--------|---------|
| **MFCoreWithTopicsScorer** | Primary scorer.  Matrix factorization on all notes, with topic-aware parameters. |
| **MFCoreScorer** | Same algorithm without topic awareness.  Provides a baseline. |
| **MFExpansionScorer** | Scores notes from "expansion" raters (newer/less established contributors). |
| **MFExpansionPlusScorer** | Extended expansion scoring with additional rater groups. |
| **ReputationScorer** | Reputation-based matrix factorization variant. |
| **MFGroupScorer** (×15+) | One scorer per modeling group.  Each group represents a cluster of raters with similar perspectives.  Allows bridging across viewpoints. |
| **MFTopicScorer** (×many) | One scorer per topic.  Scores notes within a single topic using only raters active in that topic. |
| **MFMultiGroupScorer** | Scores across multiple groups simultaneously. |

Each scorer's `prescore()` does the following internally:

1. **Initial MF** — run matrix factorization on the full rating matrix to learn a 1-dimensional embedding for each note (intercept + factor) and each rater (intercept + factor).
2. **Tentative status assignment** — apply scoring rules (CRH/CRNH/NMR thresholds) to get preliminary note statuses.  These statuses are **not** the final output — they exist only to evaluate raters in the next step.
3. **Rater helpfulness scoring** — using the tentative statuses, determine which raters are "helpful" (their ratings tend to agree with the consensus).  This includes harassment tag consensus filtering.
4. **Helpfulness-filtered MF** — re-run matrix factorization using only ratings from raters deemed helpful.  This produces refined note and rater parameters.
5. **Low-diligence model** — a secondary MF that identifies notes/raters where "incorrect" tags are used with low diligence.
6. **Tag filter thresholds** — compute percentile thresholds for not-helpful tags, used during final scoring.

#### 2d. Combine Prescoring Results

The outputs from all scorers are combined into two unified DataFrames:

- **prescoringNoteModelOutput** — one row per (note, scorer) with MF parameters.
- **prescoringRaterModelOutput** — one row per (rater, scorer) with MF parameters and helpfulness scores.

Also produces **prescoringMetaOutput** containing per-scorer metadata (global intercept, tag thresholds, helpfulness thresholds, final-round counts).

#### 2e. PFlip+ Classifier Training

A classifier is trained to predict whether a note is likely to "flip" status (CRH → NMR or vice versa) in the near future.  Used during final scoring to stabilize recently-changed notes.

### Stage 3: Final Scoring (`run_final_note_scoring`)

Final scoring produces the actual published note statuses.  It uses prescoring's output as initialization rather than starting from scratch.

#### 3a. PFlip Prediction

For notes currently in stabilization (recently changed status), predict flip probability using the PFlip+ classifier trained in prescoring.

#### 3b. Determine Which Notes to Rescore

In production (incremental mode), not all notes are rescored every run.  The system identifies notes that need rescoring:
- Notes with new ratings since the last run
- Notes that flipped status recently
- Notes not rescored recently enough
- Notes eligible for status locking

In full-run mode (no previous scores), all notes are scored.

#### 3c. Preprocessing and Filtering

- Recompute `helpfulNum` from raw helpful/notHelpful columns.
- Assign topics using the classifier trained in prescoring.
- Apply PSS filtering to remove manipulative raters' ratings.

#### 3d. Scorer Final Calls

Each scorer runs its `score_final()` method, which differs significantly from prescoring:

1. **Subsidiary MF runs** — three additional MF runs to measure robustness:
   - **No-high-volume**: exclude high-volume raters → get intercept.
   - **No-correlated**: exclude correlated raters → get intercept.
   - **Population-sampled**: use only population-sampled ratings → get intercept.
   These subsidiary runs use `interceptOnly` mode — they only run MF and return the note intercept, skipping the expensive downstream steps.

2. **Main MF run** — run helpfulness-filtered MF initialized from prescoring parameters, with rater parameters **frozen** (only note parameters are learned).  The subsidiary intercepts are merged into the note parameters.

3. **Pseudoraters** — add synthetic raters with extreme parameters and re-run MF to estimate upper/lower confidence bounds on note intercepts.

4. **Low-diligence model** — fit the diligence model on final-round ratings.

5. **`compute_scored_notes`** (with `finalRound=True`) — the full scoring rules pipeline:
   - Compute per-note rating statistics (count, helpfulness ratio, etc.).
   - Compute per-note tag aggregates weighted by rater-note embedding distance.
   - Compute incorrect-tag aggregates for the incorrect filter.
   - Apply scoring rules in priority order to assign CRH/CRNH/NMR:
     - Base CRH/CRNH thresholds on intercept
     - UCB-based CRNH
     - Ratio-based CRNH
     - Non-misleading CRNH
     - **Tag outlier filtering** — demote CRH notes with outlier not-helpful tag patterns
     - **Incorrect filtering** — demote CRH notes with high "incorrect" tag rates
     - **CRH inertia** — notes that were CRH get a slightly lower threshold to maintain status (sticky scoring)
     - **CRH super-threshold** — override tag filtering for notes with very high intercepts
     - **Low-diligence filter** — check for low-diligence note intercept
     - **Firm reject** — strong CRNH for notes far below threshold
     - **No-high-vol / no-correlated checks** — require CRH even when high-volume or correlated raters are excluded
     - **Minority net-helpful** — require minimum support from minority-viewpoint raters

6. **Helpfulness scores** — compute final rater helpfulness scores using prescoring's parameters merged with final MF output.

#### 3e. Combine Final Scorer Results

Merge results from all scorers into unified `scoredNotes` and `auxiliaryNoteInfo` DataFrames.  Merge in PFlip predictions.

### Stage 4: Post-Scoring (`post_note_scoring`)

After individual scorers finish, several cross-scorer operations run:

1. **Compute note stats** — aggregate rating statistics independent of any scorer (total ratings, helpfulness ratio over time windows, etc.).

2. **Meta-scoring** — the final arbiter that combines all individual scorer outputs to determine a single published status for each note:
   - Determines which scorer's output to use as the "deciding" status (core, expansion, group, topic).
   - Applies status locking (notes that have been CRH long enough get locked and won't change).
   - Applies NMR-due-to-min-stable-CRH-time (new CRH notes must maintain status for a minimum period).
   - Assigns the final `ratingStatus` and `decidedBy` fields.

3. **Flip checking** — validates that the rate of status changes between this run and the previous run is within acceptable bounds.  If too many notes flip, the run may be rejected.

4. **Update note status history** — merge new statuses into the status history, recording timestamps.

### Stage 5: Contributor Scoring (`run_contributor_scoring`)

After note statuses are finalized, compute per-contributor helpfulness scores:

1. **Coalesce helpfulness scores** — merge helpfulness scores from individual scorers (core, group, multi-group) into unified contributor-level scores.
2. **Compute contribution stats** — using the final note statuses, compute each contributor's track record:
   - Number of notes written that achieved CRH/CRNH.
   - Rating accuracy (how often they rated in agreement with the final consensus).
   - Success/failure counts for helpful and not-helpful ratings.
   - Aggregate ratio metrics used for enrollment decisions.

## Output Data

The pipeline produces four output files:

- **scored_notes.tsv** — one row per note with final status (`CURRENTLY_RATED_HELPFUL`, `CURRENTLY_RATED_NOT_HELPFUL`, `NEEDS_MORE_RATINGS`), MF parameters, scoring rule details, and which scorer decided the status.
- **helpfulness_scores.tsv** — one row per contributor with helpfulness scores, rating accuracy, and enrollment-relevant metrics.
- **note_status_history.tsv** — updated status history with new timestamps.
- **aux_note_info.tsv** — per-note tag aggregates, adjusted ratios, and auxiliary scoring data.

## Caching

Two dev-cache checkpoints exist for faster iteration:

1. **`runner_data`** — caches the loaded and preprocessed input DataFrames (after ID factorization and sampling). Skips TSV parsing on repeat runs.
2. **`post_prescoring`** — caches all six prescoring outputs. Skips rater clustering, topic model training, PSS filtering, and all scorer prescore() calls — jumping straight to final scoring.
