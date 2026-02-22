# Std libraries
import gc
import logging
from typing import Set, Tuple

# Project libraries
from . import constants as c

# 3rd-party libraries
import numpy as np
import pandas as pd
from numba import njit, int64


logger = logging.getLogger("birdwatch.quasi_clique_detection")
logger.setLevel(logging.INFO)


@njit(cache=True)
def _fill_group_pairs(raters, group_starts, group_tweet_ids, n_groups, n_total,
                      out_tweets, out_left, out_right):
  """Emit all C(k,2) pairs per group into pre-allocated arrays."""
  pos = int64(0)
  for g in range(n_groups):
    start = group_starts[g]
    end = group_starts[g + 1] if g + 1 < n_groups else n_total
    if end - start < 2:
      continue
    tweet = group_tweet_ids[g]
    for i in range(start, end):
      for j in range(i + 1, end):
        out_tweets[pos] = tweet
        if raters[i] < raters[j]:
          out_left[pos] = raters[i]
          out_right[pos] = raters[j]
        else:
          out_left[pos] = raters[j]
          out_right[pos] = raters[i]
        pos += 1
  return pos


class QuasiCliqueDetection:
  def __init__(
    self,
    recencyCutoff: int = (1000 * 60 * 60 * 24 * 7 * 13),
    noteInclusionThreshold: float = 0.25,
    raterInclusionThreshold: float = 0.25,
    minCliqueTweets: int = 25,
    minCliqueRaters: int = 5,
    maxCliqueSize: int = 2000,
    minInclusionRatings: int = 4,
    minRaterPairCount: int = 50,
  ):
    """Initialize QuasiCliqueDetection.

    Args:
      recencyCutoff:
      noteInclusionThreshold: At least noteInclusionThreshold must rate a given note in the same way for the note to be included
      raterInclusionThreshold: Each rater must rate at least this fraction of posts in the common way for the rater to be included
      minCliqueTweets: Each quasi-clique must have at least this many posts
      minCliqueRaters: Each clique must have at least this many raters.
      maxCliqueSize: Each quasi-clique can have at most this many raters
      minInclusionRatings: In addition to meeting the noteInclusionThreshold, included notes must get at
        least this many ratings from included raters (or be rated by all included raters, whichever is
        fewer)
      minRaterPairCount: Raters must have at least this many matching ratings (roughly 1/day) to be considered
    """
    self._recencyCutoff = recencyCutoff
    self._minCliqueTweets = minCliqueTweets
    self._minCliqueRaters = minCliqueRaters
    self._maxCliqueSize = maxCliqueSize
    self._noteInclusionThreshold = noteInclusionThreshold
    self._raterInclusionThreshold = raterInclusionThreshold
    self._minInclusionRatings = minInclusionRatings
    self._minRaterPairCount = minRaterPairCount

  def _get_pair_counts(
    self,
    ratings: pd.DataFrame,
    notes: pd.DataFrame,
    cutoff: int,
    minAlignedRatings: int = 5,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes counts of how many times raters rate notes in the same way, and all ratings for raters who do so >5 times.
    
    Returns:
      - counts: DataFrame with columns "left", "right", "count"
      - ratings: DataFrame with ratings for raters who do so >5 times
    """
    # Identify ratings that are in scope
    logger.info(f"initial rating length: {len(ratings)}")

    # this merge take 1.5 minutes
    ratings = ratings[[c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]]
    ratings = ratings.merge(
      notes[[c.noteIdKey, c.tweetIdKey, c.classificationKey, c.createdAtMillisKey]].rename(
        columns={c.createdAtMillisKey: "noteMillis"}
      )
    )
    logger.info(f"ratings after merges: {len(ratings)}")
    ratings = ratings[ratings["noteMillis"] > (ratings["noteMillis"].max() - cutoff)]
    logger.info(f"recent ratings: {len(ratings)}")
    ratings = ratings[ratings[c.tweetIdKey] != "-1"]
    logger.info(f"ratings on non-deleted tweets: {len(ratings)}")
    ratings = ratings[ratings[c.classificationKey] == c.notesSaysTweetIsMisleadingKey]
    logger.info(f"ratings on misleading posts: {len(ratings)}")
    # Identify pairs using Numba + array-based dedup (same approach as PSS)
    ratings = ratings[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey, c.raterParticipantIdKey]]

    # Factorize IDs to int32 codes
    rater_codes, rater_uniques = pd.factorize(ratings[c.raterParticipantIdKey])
    rater_arr = rater_codes.astype(np.int32)
    del rater_codes

    tweet_codes, _ = pd.factorize(ratings[c.tweetIdKey])
    tweet_arr = tweet_codes.astype(np.int32)
    n_tweets = int(tweet_codes.max()) + 1
    del tweet_codes

    note_codes, _ = pd.factorize(ratings[c.noteIdKey])
    note_arr = note_codes.astype(np.int32)
    del note_codes

    helpful_codes, _ = pd.factorize(ratings[c.helpfulNumKey])
    helpful_arr = helpful_codes.astype(np.int32)
    del helpful_codes

    # Sort by (note, helpfulNum) for contiguous group access
    sort_idx = np.lexsort((helpful_arr, note_arr))
    tweet_arr = tweet_arr[sort_idx]
    note_arr = note_arr[sort_idx]
    helpful_arr = helpful_arr[sort_idx]
    rater_arr = rater_arr[sort_idx]
    del sort_idx

    # Compute (note, helpfulNum) group boundaries
    n = len(rater_arr)
    group_mask = np.empty(n, dtype=np.bool_)
    group_mask[0] = True
    group_mask[1:] = (note_arr[1:] != note_arr[:-1]) | (helpful_arr[1:] != helpful_arr[:-1])
    group_starts = np.where(group_mask)[0].astype(np.int64)
    n_groups = len(group_starts)
    group_tweet_ids = tweet_arr[group_starts]
    n_raters = len(rater_uniques)
    del tweet_arr, note_arr, helpful_arr, group_mask
    gc.collect()

    # Compute total pair events from group sizes
    group_sizes = np.diff(np.append(group_starts, n)).astype(np.int64)
    total_events = int(np.sum(group_sizes * (group_sizes - 1) // 2))
    del group_sizes
    logger.info(f"Total pair events: {total_events:,} across {n_groups:,} groups")

    if total_events == 0:
      return pd.DataFrame(columns=["left", "right", "count"]), ratings.iloc[:0]

    # Fill pair arrays with Numba
    out_tweets = np.empty(total_events, dtype=np.int32)
    out_left = np.empty(total_events, dtype=np.int32)
    out_right = np.empty(total_events, dtype=np.int32)
    _fill_group_pairs(
      rater_arr, group_starts, group_tweet_ids,
      int64(n_groups), int64(n),
      out_tweets, out_left, out_right,
    )
    del rater_arr, group_starts, group_tweet_ids
    gc.collect()

    # Pack into single int64, sort, dedup per tweet, count per pair
    rater_bits = max(1, int(np.ceil(np.log2(max(n_raters, 2)))))
    tweet_bits = max(1, int(np.ceil(np.log2(max(n_tweets, 2)))))
    assert 2 * rater_bits + tweet_bits <= 63
    left_shift = rater_bits + tweet_bits
    right_shift = tweet_bits

    compound = (
      (out_left.astype(np.int64) << left_shift) |
      (out_right.astype(np.int64) << right_shift) |
      out_tweets.astype(np.int64)
    )
    del out_left, out_right, out_tweets
    gc.collect()

    compound.sort()

    # Dedup per tweet
    dup_mask = np.empty(len(compound), dtype=np.bool_)
    dup_mask[0] = True
    dup_mask[1:] = compound[1:] != compound[:-1]
    compound = compound[dup_mask]
    del dup_mask
    gc.collect()

    # Count per pair via run-length encoding
    pair_key = compound >> tweet_bits
    pair_boundary = np.empty(len(pair_key), dtype=np.bool_)
    pair_boundary[0] = True
    pair_boundary[1:] = pair_key[1:] != pair_key[:-1]
    pair_starts = np.where(pair_boundary)[0]
    pair_counts_arr = np.diff(np.append(pair_starts, len(pair_key)))
    del pair_key, pair_boundary
    gc.collect()

    # Filter by minAlignedRatings and extract rater codes
    count_mask = pair_counts_arr >= minAlignedRatings
    surviving = compound[pair_starts[count_mask]]
    surviving_counts = pair_counts_arr[count_mask]
    del compound, pair_starts, pair_counts_arr, count_mask
    gc.collect()

    rater_mask = (1 << rater_bits) - 1
    result_left_codes = ((surviving >> left_shift) & rater_mask).astype(np.int32)
    result_right_codes = ((surviving >> right_shift) & rater_mask).astype(np.int32)

    # Map back to original rater IDs
    result_left = rater_uniques[result_left_codes]
    result_right = rater_uniques[result_right_codes]

    counts = pd.DataFrame({
      "left": result_left,
      "right": result_right,
      "count": surviving_counts.astype(int),
    })

    # Filter ratings to raters that appear in the pair counts
    included_raters = set(result_left) | set(result_right)
    ratings = ratings[ratings[c.raterParticipantIdKey].isin(included_raters)]
    logger.info(f"ratings after filter to raters included in pair counts: {len(ratings)}")
    return counts, ratings

  def _prepare_grow_clique_arrays(self, raterRatings: pd.DataFrame) -> dict:
    """Convert raterRatings DataFrame to numpy arrays and CSR inverted indices.

    Called once before the clique-building loop. The returned arrays are reused
    across all _grow_clique calls, eliminating all pandas operations from the
    inner loop.

    Builds:
      - Factorized int32 codes for raters, actions (tweet+note+helpful triples), and tweets
      - CSR-style rater->actions and action->raters inverted indices
      - Action-to-tweet mapping for fast unique-tweet counting
      - Reverse lookup dict from original rater IDs to integer codes
    """
    rater_col = raterRatings[c.raterParticipantIdKey].values
    tweet_col = raterRatings[c.tweetIdKey].values
    note_col = raterRatings[c.noteIdKey].values
    helpful_col = raterRatings[c.helpfulNumKey].values

    # Factorize rater and tweet IDs to contiguous int32 codes
    rater_codes, rater_uniques = pd.factorize(rater_col)
    rater_codes = rater_codes.astype(np.int32)
    n_raters = len(rater_uniques)

    tweet_codes, tweet_uniques = pd.factorize(tweet_col)
    tweet_codes = tweet_codes.astype(np.int32)
    n_tweets = len(tweet_uniques)

    # Build compound action key from (tweet, note, helpfulNum) using bit-packing and factorize
    note_codes, _ = pd.factorize(note_col)
    helpful_codes, _ = pd.factorize(helpful_col)
    n_notes_val = max(int(note_codes.max()) + 1, 2) if len(note_codes) > 0 else 2
    n_helpful_val = max(int(helpful_codes.max()) + 1, 2) if len(helpful_codes) > 0 else 2
    note_bits = max(1, int(np.ceil(np.log2(n_notes_val))))
    helpful_bits = max(1, int(np.ceil(np.log2(n_helpful_val))))

    action_compound = (
      (tweet_codes.astype(np.int64) << (note_bits + helpful_bits))
      | (note_codes.astype(np.int64) << helpful_bits)
      | helpful_codes.astype(np.int64)
    )
    action_codes, action_compound_uniques = pd.factorize(action_compound)
    action_codes = action_codes.astype(np.int32)
    n_actions = len(action_compound_uniques)

    # Map each action code back to its tweet code
    tweet_of_action = (action_compound_uniques >> (note_bits + helpful_bits)).astype(np.int32)
    del action_compound, action_compound_uniques, note_codes, helpful_codes

    # Deduplicate (rater, action) pairs (safety; should already be unique after preprocessing)
    action_bits = max(1, int(np.ceil(np.log2(max(n_actions, 2)))))
    pair_key = (rater_codes.astype(np.int64) << action_bits) | action_codes.astype(np.int64)
    _, uniq_idx = np.unique(pair_key, return_index=True)
    rater_codes = rater_codes[uniq_idx]
    action_codes = action_codes[uniq_idx]
    del pair_key, uniq_idx

    # Build rater -> actions CSR index
    sort_r = np.argsort(rater_codes, kind='mergesort')
    rat_act_data = action_codes[sort_r].copy()
    rat_counts = np.bincount(rater_codes, minlength=n_raters)
    rat_act_indptr = np.empty(n_raters + 1, dtype=np.int64)
    rat_act_indptr[0] = 0
    np.cumsum(rat_counts, out=rat_act_indptr[1:])
    del sort_r, rat_counts

    # Build action -> raters CSR index
    sort_a = np.argsort(action_codes, kind='mergesort')
    act_rat_data = rater_codes[sort_a].copy()
    act_counts = np.bincount(action_codes, minlength=n_actions)
    act_rat_indptr = np.empty(n_actions + 1, dtype=np.int64)
    act_rat_indptr[0] = 0
    np.cumsum(act_counts, out=act_rat_indptr[1:])
    del sort_a, act_counts, rater_codes, action_codes

    rater_id_to_code = {rid: i for i, rid in enumerate(rater_uniques)}

    logger.info(
      f"Prepared grow-clique arrays: {n_raters:,} raters, {n_actions:,} actions, "
      f"{n_tweets:,} tweets, {len(rat_act_data):,} unique (rater,action) pairs"
    )
    return {
      'n_raters': n_raters,
      'n_actions': n_actions,
      'n_tweets': n_tweets,
      'rater_uniques': rater_uniques,
      'tweet_uniques': tweet_uniques,
      'tweet_of_action': tweet_of_action,
      'rat_act_data': rat_act_data,
      'rat_act_indptr': rat_act_indptr,
      'act_rat_data': act_rat_data,
      'act_rat_indptr': act_rat_indptr,
      'rater_id_to_code': rater_id_to_code,
    }

  def _grow_clique(
    self,
    seedCodes: Set[int],
    arrays: dict,
  ):
    """Grow a clique from seed rater codes using pre-built numpy arrays.

    Equivalent to the original pandas-based _grow_clique but operates entirely on
    integer-coded numpy arrays and CSR inverted indices.  Action counts are maintained
    incrementally — O(k) per rater added (k = rater's actions) — instead of recomputing
    value_counts over the full DataFrame each iteration.

    Args:
      seedCodes: Set of integer rater codes to seed the clique
      arrays: Dict of pre-built arrays from _prepare_grow_clique_arrays

    Returns:
      (set of original rater IDs, set of original tweet IDs) meeting density criteria.
    """
    n_raters = arrays['n_raters']
    n_actions = arrays['n_actions']
    n_tweets = arrays['n_tweets']
    tweet_of_action = arrays['tweet_of_action']
    rat_act_data = arrays['rat_act_data']
    rat_act_indptr = arrays['rat_act_indptr']
    act_rat_data = arrays['act_rat_data']
    act_rat_indptr = arrays['act_rat_indptr']
    rater_uniques = arrays['rater_uniques']
    tweet_uniques = arrays['tweet_uniques']

    # --- Initialize state ---
    included = np.zeros(n_raters, dtype=np.bool_)
    for r in seedCodes:
      included[r] = True
    n_included = len(seedCodes)

    # Incremental per-action count of how many included raters have each action
    action_count = np.zeros(n_actions, dtype=np.int32)
    for r in seedCodes:
      s, e = rat_act_indptr[r], rat_act_indptr[r + 1]
      action_count[rat_act_data[s:e]] += 1

    saved_tweet_codes = np.empty(0, dtype=np.int32)

    for _ in range(self._maxCliqueSize):
      # --- Step 1: Find qualifying actions (both thresholds must be met) ---
      is_qualifying = (
        (action_count >= n_included * self._noteInclusionThreshold)
        & (action_count >= min(n_included, self._minInclusionRatings))
      )
      qual_actions = np.where(is_qualifying)[0]
      if len(qual_actions) == 0:
        break

      # --- Step 2: Find best candidate (non-included rater with most aligned unique tweets) ---
      # Gather all (rater, tweet) pairs from qualifying actions via the action->raters CSR index
      starts = act_rat_indptr[qual_actions]
      ends = act_rat_indptr[qual_actions + 1]
      sizes = (ends - starts).astype(np.int64)
      total_size = int(sizes.sum())
      if total_size == 0:
        break

      all_r = np.empty(total_size, dtype=np.int32)
      all_t = np.empty(total_size, dtype=np.int32)
      pos = 0
      for i in range(len(qual_actions)):
        sz = int(sizes[i])
        if sz > 0:
          all_r[pos:pos + sz] = act_rat_data[starts[i]:ends[i]]
          all_t[pos:pos + sz] = tweet_of_action[qual_actions[i]]
          pos += sz

      # Filter to non-included raters only
      mask = ~included[all_r]
      cand_r = all_r[mask]
      cand_t = all_t[mask]
      del all_r, all_t, mask

      if len(cand_r) == 0:
        break

      # Deduplicate (rater, tweet) pairs and count unique tweets per candidate rater
      tweet_bits = max(1, int(np.ceil(np.log2(max(n_tweets, 2)))))
      compound = (cand_r.astype(np.int64) << tweet_bits) | cand_t.astype(np.int64)
      compound = np.unique(compound)
      r_of_compound = (compound >> tweet_bits).astype(np.int32)
      scores = np.bincount(r_of_compound, minlength=n_raters)
      scores[included] = -1
      candidate = int(np.argmax(scores))
      del compound, r_of_compound, scores, cand_r, cand_t

      # --- Step 3: Save current qualifying tweets (before trial add) ---
      saved_tweet_codes = np.unique(tweet_of_action[qual_actions])

      # --- Step 4: Trial-add the candidate ---
      cand_s, cand_e = rat_act_indptr[candidate], rat_act_indptr[candidate + 1]
      cand_actions = rat_act_data[cand_s:cand_e]
      action_count[cand_actions] += 1
      n_trial = n_included + 1

      # Recompute qualifying actions with updated counts and thresholds
      is_qual_trial = (
        (action_count >= n_trial * self._noteInclusionThreshold)
        & (action_count >= min(n_trial, self._minInclusionRatings))
      )
      qual_trial = np.where(is_qual_trial)[0]
      trial_tweet_codes = np.unique(tweet_of_action[qual_trial])
      satisfiedTweets = len(trial_tweet_codes)

      # --- Step 5: Reject if not enough tweets ---
      if satisfiedTweets < self._minCliqueTweets:
        action_count[cand_actions] -= 1
        included_codes = np.where(included)[0]
        return set(rater_uniques[included_codes]), set(tweet_uniques[saved_tweet_codes])

      # --- Step 6: Reject if any rater falls below the inclusion threshold ---
      rater_threshold = self._raterInclusionThreshold * satisfiedTweets
      raters_to_check = np.append(np.where(included)[0], candidate)
      for r in raters_to_check:
        r_qual = rat_act_data[rat_act_indptr[r]:rat_act_indptr[r + 1]]
        r_qual = r_qual[is_qual_trial[r_qual]]
        if len(r_qual) > 0 and len(np.unique(tweet_of_action[r_qual])) < rater_threshold:
          action_count[cand_actions] -= 1
          included_codes = np.where(included)[0]
          return set(rater_uniques[included_codes]), set(tweet_uniques[saved_tweet_codes])

      # --- Step 7: Accept the candidate ---
      included[candidate] = True
      n_included += 1

    # Loop exhausted (max iterations or no qualifying actions/candidates)
    included_codes = np.where(included)[0]
    return set(rater_uniques[included_codes]), set(tweet_uniques[saved_tweet_codes])

  def _build_clusters(
    self,
    raterPairCounts: pd.DataFrame,
    raterPairRatings: pd.DataFrame,
  ):
    """Identify disjoint quasi-cliques using a greedy clustering approach.

    Args:
      raterPairCounts: DF containing counts of how often pairs of raters colide.
      raterPairRatings: All ratings from raters with >5 collisions with another rater
    """
    # Pre-build numpy arrays and inverted indices once for all grow_clique calls
    arrays = self._prepare_grow_clique_arrays(raterPairRatings)

    cliques = []
    # Attempt to cluster every rater with at least minRaterPairCount collisions
    logger.info(f"orig raterPairCounts: {len(raterPairCounts)}")
    raterPairCounts = raterPairCounts[raterPairCounts["count"] > self._minRaterPairCount]
    logger.info(f"pruned raterPairCounts: {len(raterPairCounts)}")
    # Build cliques
    while len(raterPairCounts) > 0:
      # Identify seed
      logger.info(f"sorting raterPairCounts")
      raterPairCounts = raterPairCounts.sort_values("count", ascending=False)
      leftRater, rightRater = raterPairCounts.head(1)[["left", "right"]].values.flatten()
      # Convert seed rater IDs to integer codes and grow clique
      leftCode = arrays['rater_id_to_code'][leftRater]
      rightCode = arrays['rater_id_to_code'][rightRater]
      logger.info(f"growing clique with")
      cliqueRaters, cliquePosts = self._grow_clique({leftCode, rightCode}, arrays)
      logger.info(f"pruning raterPairCounts {len(raterPairCounts)}")
      raterPairCounts = raterPairCounts[
        ~(
          (raterPairCounts["left"].isin(cliqueRaters))
          | (raterPairCounts["right"].isin(cliqueRaters))
        )
      ]
      # Augment results if clique is large enough
      if len(cliqueRaters) >= self._minCliqueRaters:
        logger.info(
          f"Adding clique  (raters={len(cliqueRaters)}, tweets={len(cliquePosts)}).  Remaining ratePairCounts: {len(raterPairCounts)}"
        )
        cliques.append((cliqueRaters, cliquePosts))
      else:
        logger.info(
          f"Skipping clique  (raters={len(cliqueRaters)}, tweets={len(cliquePosts)}).  Remaining ratePairCounts: {len(raterPairCounts)}"
        )
    # Order from largest to smallest and return
    cliques.sort(key=lambda clique: len(clique[0]))
    return cliques

  def get_quasi_cliques(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
  ) -> pd.DataFrame:
    """Obtain quasi-cliques in the rating graph.

    Each clique is defined by a list of raters and posts, and must meet minimum
    size and density requirements with respect to the number of raters, number of
    posts and minium density of ratings connecting the raters and posts.
    """
    # Obtain quasi-cliques
    raterPairCounts, raterPairRatings = self._get_pair_counts(ratings, notes, self._recencyCutoff)
    quasiCliques = self._build_clusters(raterPairCounts, raterPairRatings)
    # Convert to data frame
    cliqueIds = []
    raterIds = []
    for i, (raters, _) in enumerate(quasiCliques):
      for rater in raters:
        cliqueIds.append((i + 1))  # To align with PSS, by convention clique IDs begin at 1
        raterIds.append(rater)
    return pd.DataFrame(
      {
        c.raterParticipantIdKey: pd.array(raterIds, dtype=np.int64),
        c.quasiCliqueValueKey: pd.array(cliqueIds, dtype=pd.Int64Dtype()),
      }
    )
