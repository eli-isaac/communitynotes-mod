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
    """Return counts of how many times raters rate notes in the same way, and all ratings for raters who do so >5 times."""
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

  def _grow_clique(
    self,
    includedRaters: Set[str],
    raterRatings: pd.DataFrame,
  ):
    """Grow a clique from an initial set of raters.  Return all included raters and tweets.

    Given a clique seed containing two raters, greedily grow the clique one rater at a time
    by adding the excluded rater that has the most ratings in common with the actions of the
    clique.  Before each addition, verify that adding the rater will not violate the minimum
    density requirements of the clique.

    Args:
      includedRaters: Set of raters to use to initialize a clique
      raterRatings: DF containing ratings from all raters with 5 or more rating collisions

    Returns:
      Set of raters and tweets that meet density criteria.
    """
    # Expand cliques 1 rater at a time, checking to see if density thresholds are satisifed after expansion
    for _ in range(self._maxCliqueSize):
      # Identify the ratings where there is enough agreement that the rating constitutes a group action
      ratings = raterRatings[raterRatings[c.raterParticipantIdKey].isin(includedRaters)]
      groupRatingCounts = (
        ratings[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey]].value_counts().reset_index(drop=False)
      )
      groupRatingCounts = groupRatingCounts[
        groupRatingCounts["count"] >= (len(includedRaters) * self._noteInclusionThreshold)
      ]
      groupRatingCounts = groupRatingCounts[
        groupRatingCounts["count"] >= min(len(includedRaters), self._minInclusionRatings)
      ]
      # Find the rater not in the group that most aligned with the group rating events
      alignedRatings = raterRatings.merge(
        groupRatingCounts[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey]]
      )
      alignedRatings = alignedRatings[~alignedRatings[c.raterParticipantIdKey].isin(includedRaters)]
      alignedTweetPerRater = (
        alignedRatings[[c.tweetIdKey, c.raterParticipantIdKey]]
        .drop_duplicates()[c.raterParticipantIdKey]
        .value_counts()
        .reset_index(drop=False)
      )
      candidate = (
        alignedTweetPerRater.sort_values("count", ascending=False)
        .head(1)[c.raterParticipantIdKey]
        .item()
      )
      # Preserve current set of tweets incase we decide not to add the candidate
      includedTweets = set(groupRatingCounts[c.tweetIdKey].drop_duplicates())
      # Calculate how many tweets would meet the inclusion threshold if the candidate were added
      candidateRaters = includedRaters | {candidate}
      ratings = raterRatings[raterRatings[c.raterParticipantIdKey].isin(candidateRaters)]
      groupRatingCounts = (
        ratings[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey]].value_counts().reset_index(drop=False)
      )
      groupRatingCounts = groupRatingCounts[
        groupRatingCounts["count"] >= (len(candidateRaters) * self._noteInclusionThreshold)
      ]
      groupRatingCounts = groupRatingCounts[
        groupRatingCounts["count"] >= min(len(candidateRaters), self._minInclusionRatings)
      ]
      satisfiedTweets = groupRatingCounts[c.tweetIdKey].nunique()
      # Calculate how many raters would be below the inclusion threshold if we add the candidate
      matchingRatings = ratings.merge(
        groupRatingCounts[[c.tweetIdKey, c.noteIdKey, c.helpfulNumKey]]
      )
      raterCounts = (
        matchingRatings[[c.raterParticipantIdKey, c.tweetIdKey]]
        .drop_duplicates()
        .value_counts(c.raterParticipantIdKey)
        .reset_index(drop=False)
      )
      ratersBelowThreshold = (
        raterCounts["count"] < (self._raterInclusionThreshold * satisfiedTweets)
      ).sum()
      # Check standards
      if satisfiedTweets >= self._minCliqueTweets and ratersBelowThreshold == 0:
        includedRaters = candidateRaters
      else:
        return includedRaters, includedTweets
    return includedRaters, includedTweets

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
    cliques = []
    # Attempt to cluster every rater with at least minRaterPairCount collisions
    logger.info(f"orig raterPairCounts: {len(raterPairCounts)}")
    raterPairCounts = raterPairCounts[raterPairCounts["count"] > self._minRaterPairCount]
    logger.info(f"pruned raterPairCounts: {len(raterPairCounts)}")
    # Build cliques
    while len(raterPairCounts) > 0:
      # Identify seed
      raterPairCounts = raterPairCounts.sort_values("count", ascending=False)
      leftRater, rightRater = raterPairCounts.head(1)[["left", "right"]].values.flatten()
      # Build clique and prune candidate set
      cliqueRaters, cliquePosts = self._grow_clique({leftRater, rightRater}, raterPairRatings)
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
        c.raterParticipantIdKey: raterIds,
        c.quasiCliqueValueKey: cliqueIds,
      }
    )
