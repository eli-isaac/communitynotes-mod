import gc
import logging
import sys
from typing import Dict

from . import constants as c, dev_cache

import numpy as np
import pandas as pd
from numba import njit, int64


logger = logging.getLogger("birdwatch.post_selection_similarity")


def _mem_gb():
  """Current process RSS in GB. Falls back to max RSS if /proc unavailable."""
  try:
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) / (1024 * 1024)
  except FileNotFoundError:
    pass
  import resource
  ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return ru / (1024 ** 3) if sys.platform == "darwin" else ru / (1024 * 1024)


def _mem_available_gb():
  """Available system memory in GB, or None if unknown."""
  try:
    with open("/proc/meminfo") as f:
      for line in f:
        if line.startswith("MemAvailable:"):
          return int(line.split()[1]) / (1024 * 1024)
  except FileNotFoundError:
    return None
logger.setLevel(logging.INFO)


class PostSelectionSimilarity:
  def __init__(
    self,
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    pmiRegularization: int = 500,
    smoothedNpmiThreshold: float = 0.55,
    minimumRatingProportionThreshold: float = 0.4,
    minUniquePosts: int = 10,
    minSimPseudocounts: int = 10,
    windowMillis: int = 1000 * 60 * 20,
  ):
    # Try loading cached state computed before _get_pair_counts_dict.
    # When the cache hits, everything above _get_pair_counts_dict is skipped.
    cached = dev_cache.load("pss_pre_pair_counts")
    if cached is not None:
      self.affinityAndCoverage = cached["affinityAndCoverage"]
      self.suspectPairs = cached["suspectPairs"]
      self.ratings = cached["ratings"]
      logger.info("Restored PSS intermediate state from cache — skipping to _get_pair_counts_dict")
    else:
      # Compute rater affinity and writer coverage.  Apply thresholds to identify linked pairs.
      logger.info("Computing rater affinity and writer coverage")
      # this takes 14 seconds to run
      helpful_mask = ratings[c.helpfulnessLevelKey] == c.helpfulValueTsv
      helpfulRatings = ratings.loc[helpful_mask, [c.raterParticipantIdKey, c.noteIdKey, c.createdAtMillisKey]]
      logger.info(f"Computing rater affinity and writer coverage for {len(helpfulRatings)} helpful ratings")
      # this takes 3.5 minutes to run
      self.affinityAndCoverage = self.compute_affinity_and_coverage(helpfulRatings, notes, [1, 5, 20])
      logger.info(f"Computed rater affinity and writer coverage for {len(self.affinityAndCoverage)} pairs")

      # runs quickly
      self.suspectPairs = self.get_suspect_pairs(self.affinityAndCoverage)
      logger.info(f"Found {len(self.suspectPairs)} suspect pairs")

      # Compute MinSim and NPMI
      # this takes 2.5 minutes to run
      self.ratings = _preprocess_ratings(notes, ratings)
      logger.info(f"Preprocessed ratings for {len(self.ratings)} ratings")

      # Save cache so next run can skip straight to _get_pair_counts_dict.
      dev_cache.save("pss_pre_pair_counts", {
        "affinityAndCoverage": self.affinityAndCoverage,
        "suspectPairs": self.suspectPairs,
        "ratings": self.ratings,
      })

    # ---- RESUME POINT: _get_pair_counts_dict (the function being iterated on) ----
    self.uniqueRatingsOnTweets = self.ratings[
      [c.tweetIdKey, c.raterParticipantIdKey]
    ].drop_duplicates()
    logger.info(f"Computed unique ratings on tweets for {len(self.uniqueRatingsOnTweets)} ratings")
    raterTotals = self.uniqueRatingsOnTweets[c.raterParticipantIdKey].value_counts()
    raterTotalsDict = {
      index: value for index, value in raterTotals.items() if value >= minUniquePosts
    }
    N = len(self.uniqueRatingsOnTweets)

    minCoRatingCount = _min_survivable_co_rating_count(
      N, pmiRegularization, minSimPseudocounts,
      smoothedNpmiThreshold, minimumRatingProportionThreshold, minUniquePosts,
    )
    logger.info(f"Pre-filter threshold: minCoRatingCount={minCoRatingCount}")

    with c.time_block("Compute pair counts dict"):
      self.pairCountsDict = _get_pair_counts_dict(
        self.ratings, windowMillis=windowMillis, minCoRatingCount=minCoRatingCount,
      )
    logger.info(f"Computed pair counts dict for {len(self.pairCountsDict)} pairs")

    self.pairCountsDict = _join_rater_totals_compute_pmi_and_filter_edges_below_threshold(
      pairCountsDict=self.pairCountsDict,
      raterTotalsDict=raterTotalsDict,
      N=len(self.uniqueRatingsOnTweets),
      pmiPseudocounts=pmiRegularization,
      minSimPseudocounts=minSimPseudocounts,
      smoothedNpmiThreshold=smoothedNpmiThreshold,
      minimumRatingProportionThreshold=minimumRatingProportionThreshold,
    )
    self.suspectPairs = set(self.suspectPairs + list(self.pairCountsDict.keys()))
    logger.info(f"Computed suspect pairs for {len(self.suspectPairs)} pairs")
    
  def _merge_ratings_and_notes(self, ratings, notes):
    """Merge ratings with notes and compute latency. Done once, shared across all latency windows."""
    logger.info(f"Merging ratings and notes for {len(ratings)} ratings")
    ratings = ratings[[c.raterParticipantIdKey, c.noteIdKey, c.createdAtMillisKey]].rename(
      columns={c.createdAtMillisKey: "ratingMillis"}
    )
    notes = notes[[c.noteAuthorParticipantIdKey, c.noteIdKey, c.createdAtMillisKey]].rename(
      columns={c.createdAtMillisKey: "noteMillis"}
    )
    merged = ratings.merge(notes)
    merged["latency"] = merged["ratingMillis"] - merged["noteMillis"]
    logger.info(f"Merged ratings and notes: {len(merged)} rows")
    return merged

  def _compute_writer_totals(self, notes):
    """Compute total notes per author. Done once, shared across all latency windows."""
    writerTotals = (
      notes[[c.noteAuthorParticipantIdKey, c.noteIdKey, c.createdAtMillisKey]]
      .rename(columns={c.createdAtMillisKey: "noteMillis"})[c.noteAuthorParticipantIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": "writerTotal"})
    )
    logger.info(f"Computed writer totals for {len(writerTotals)} writers")
    return writerTotals

  def _compute_affinity_and_coverage_for_latency(self, merged, writerTotals, latencyMins, minDenom):
    """Compute rater affinity and writer coverage metrics for a single latency window.

    Filters the pre-merged ratings/notes data to only ratings within the given latency
    window, then computes per-pair statistics measuring how concentrated a rater's activity
    is on a particular note author (affinity) and what fraction of an author's notes a
    rater has rated (coverage). Pairs with insufficient data (below minDenom) are set to NA.

    Args:
      merged: DataFrame with columns [raterParticipantId, noteId, ratingMillis,
          noteAuthorParticipantId, noteMillis, latency]. Pre-joined ratings and notes
          produced by _merge_ratings_and_notes.
      writerTotals: DataFrame with columns [noteAuthorParticipantId, writerTotal],
          containing the total number of notes written by each author across all time
          (not filtered by latency).
      latencyMins: The latency window in minutes. Only ratings where
          (ratingMillis - noteMillis) <= latencyMins * 60 * 1000 are included.
      minDenom: Minimum number of ratings (for rater totals) or notes (for writer totals)
          required to compute a meaningful ratio. Pairs below this threshold have their
          affinity or coverage set to NA.

    Returns:
      DataFrame with columns [noteAuthorParticipantId, raterParticipantId, writerTotal,
      raterTotal{latencyMins}m, pairTotal{latencyMins}m, raterAffinity{latencyMins}m,
      writerCoverage{latencyMins}m].
    """
    logger.info(f"Computing affinity and coverage for {latencyMins}m window")
    ratings = merged[merged["latency"] <= (1000 * 60 * latencyMins)]
    logger.info(f"Filtered to {len(ratings)} ratings within {latencyMins}m")

    # Compute rater and pair totals
    raterTotals = (
      ratings[c.raterParticipantIdKey]
      .value_counts()
      .to_frame()
      .reset_index(drop=False)
      .rename(columns={"count": f"raterTotal{latencyMins}m"})
    )
    logger.info(f"Computed rater totals for {len(raterTotals)} raters")
    # rating totals takes about 7 seconds to run
    ratingTotals = (
      ratings[[c.noteAuthorParticipantIdKey, c.raterParticipantIdKey]]
      .value_counts()
      .reset_index(drop=False)
      .rename(columns={"count": f"pairTotal{latencyMins}m"})
    )
    logger.info(f"Computed pair totals for {len(ratingTotals)} pairs")
    ratingTotals = ratingTotals.merge(writerTotals, how="left")
    logger.info("Merged writer totals")
    ratingTotals = ratingTotals.merge(raterTotals, how="left")
    logger.info("Merged rater totals")

    # our df now has these columns:
    # noteAuthorParticipantId, raterParticipantId, writerTotal, raterTotal{latencyMins}m, pairTotal{latencyMins}m

    # Compute ratios
    ratingTotals[f"raterAffinity{latencyMins}m"] = (
      ratingTotals[f"pairTotal{latencyMins}m"] / ratingTotals[f"raterTotal{latencyMins}m"]
    )
    ratingTotals[f"writerCoverage{latencyMins}m"] = (
      ratingTotals[f"pairTotal{latencyMins}m"] / ratingTotals["writerTotal"]
    )
    ratingTotals.loc[
      ratingTotals[f"raterTotal{latencyMins}m"] < minDenom, f"raterAffinity{latencyMins}m"
    ] = pd.NA
    ratingTotals.loc[
      ratingTotals["writerTotal"] < minDenom, f"writerCoverage{latencyMins}m"
    ] = pd.NA
    logger.info(f"Computed affinity and coverage for {len(ratingTotals)} pairs at {latencyMins}m")
    return ratingTotals[
      [
        c.noteAuthorParticipantIdKey,
        c.raterParticipantIdKey,
        "writerTotal",
        f"raterTotal{latencyMins}m",
        f"pairTotal{latencyMins}m",
        f"raterAffinity{latencyMins}m",
        f"writerCoverage{latencyMins}m",
      ]
    ].astype(
      {
        f"raterTotal{latencyMins}m": pd.Int64Dtype(),
        f"pairTotal{latencyMins}m": pd.Int64Dtype(),
      }
    )

  def compute_affinity_and_coverage(self, ratings, notes, latencyMins, minDenom=10):
    """
    Computes rater affinity and writer coverage metrics for a list of latency windows.
    Returns a dataframe with columns [noteAuthorParticipantId, raterParticipantId, writerTotal,
    raterTotal{latencyMins}m, pairTotal{latencyMins}m, raterAffinity{latencyMins}m,
    writerCoverage{latencyMins}m] for each latency window.
    """
    latencyMins = sorted(latencyMins, reverse=True)

    # Compute shared data once, reuse for all latency windows 
    merged = self._merge_ratings_and_notes(ratings, notes)
    logger.info("Computed merged data")
    writerTotals = self._compute_writer_totals(notes)
    logger.info("Computed writer totals")

    df = self._compute_affinity_and_coverage_for_latency(merged, writerTotals, latencyMins[0], minDenom)
    logger.info("Computed affinity and coverage for 1m window")

    origLen = len(df)
    for latency in latencyMins[1:]:
      df = df.merge(
        self._compute_affinity_and_coverage_for_latency(merged, writerTotals, latency, minDenom),
        on=[c.noteAuthorParticipantIdKey, c.raterParticipantIdKey, "writerTotal"],
        how="left",
      )
      logger.info(f"Computed and merged affinity and coverage for {latency}m window")
      assert len(df) == origLen
    cols = [c.noteAuthorParticipantIdKey, c.raterParticipantIdKey, "writerTotal"]
    for latency in sorted(latencyMins):
      cols.extend(
        [
          f"raterTotal{latency}m",
          f"pairTotal{latency}m",
          f"raterAffinity{latency}m",
          f"writerCoverage{latency}m",
        ]
      )
    return df[cols]

  def get_suspect_pairs(self, affinityAndCoverage):
    """Returns a list of author-rater suspect pairs based on the affinity and coverage metrics."""
    thresholds = [
      ("writerCoverage1m", 0.2),
      ("writerCoverage5m", 0.3),
      ("writerCoverage20m", 0.4),
      ("raterAffinity1m", 0.2),
      ("raterAffinity5m", 0.45),
      ("raterAffinity20m", 0.7),
    ]
    suspectPairsDF = []
    for col, value in thresholds:
      tmp = affinityAndCoverage[affinityAndCoverage[col] >= value][
        [c.noteAuthorParticipantIdKey, c.raterParticipantIdKey]
      ].copy()
      suspectPairsDF.append(tmp)
    suspectPairsDF = pd.concat(suspectPairsDF)
    suspectPairs = []
    for author, rater in suspectPairsDF[
      [c.noteAuthorParticipantIdKey, c.raterParticipantIdKey]
    ].values:
      suspectPairs.append(tuple(sorted((author, rater))))
    return suspectPairs

  def get_post_selection_similarity_values(self):
    """
    Returns a dataframe with [raterParticipantId, postSelectionSimilarityValue] columns.
    postSelectionSimilarityValue is the ID of the suspect clique that the rater belongs to.
    There is one row for every rater that has been added to the internal suspectPairs set.
    """
    cliqueToUserMap, userToCliqueMap = aggregate_into_cliques(self.suspectPairs)
    logger.info(f"Aggregated into {len(cliqueToUserMap)} cliques")
    # Convert dict to pandas dataframe
    cliquesDfList = []
    for cliqueId in cliqueToUserMap.keys():
      for userId in cliqueToUserMap[cliqueId]:
        cliquesDfList.append({c.raterParticipantIdKey: userId, c.postSelectionValueKey: cliqueId})
    cliquesDf = pd.DataFrame(
      cliquesDfList, columns=[c.raterParticipantIdKey, c.postSelectionValueKey]
    )
    logger.info(f"Computed cliques dataframe for {len(cliquesDf)} cliques")
    return cliquesDf


def apply_post_selection_similarity(notes, ratings, postSelectionSimilarityValues):
  """
  Filters out ratings after the first on each note from raters who have high post selection similarity,
  or filters all if the note is authored by a user with the same post selection similarity value.
  """
  # Summarize input
  logger.info(f"Total ratings prior to applying post selection similarity: {len(ratings)}")
  logger.info(
    f"Total unique ratings before: {len(ratings[[c.noteIdKey, c.raterParticipantIdKey]].drop_duplicates())}"
  )
  pssSummary = (
    postSelectionSimilarityValues[[c.postSelectionValueKey, c.quasiCliqueValueKey]] > 0
  ).sum()
  logger.info(f"Summary of postSelectionSimilarityValues: \n{pssSummary}")
  # Add additional column with bit flagging correlated raters
  correlatedRaters = set(
    postSelectionSimilarityValues[postSelectionSimilarityValues[c.quasiCliqueValueKey] >= 1][
      c.raterParticipantIdKey
    ]
  )
  ratings[c.correlatedRaterKey] = ratings[c.raterParticipantIdKey].isin(correlatedRaters)
  logger.info(
    f"correlatedRater set on {ratings[c.correlatedRaterKey].sum()} ratings from {len(correlatedRaters)} unique raters"
  )
  # Trim correlated raters out of PSS dataframe and remove drop ratings flagged by NPMI/MinSim
  postSelectionSimilarityValues = (
    postSelectionSimilarityValues[[c.raterParticipantIdKey, c.postSelectionValueKey]]
    .dropna()
    .drop_duplicates()
  )
  ratingsWithPostSelectionSimilarity = (
    ratings.merge(
      postSelectionSimilarityValues,
      on=c.raterParticipantIdKey,
      how="left",
      unsafeAllowed=c.postSelectionValueKey,
    )
    .merge(notes[[c.noteIdKey, c.noteAuthorParticipantIdKey]], on=c.noteIdKey, how="left")
    .merge(
      postSelectionSimilarityValues,
      left_on=c.noteAuthorParticipantIdKey,
      right_on=c.raterParticipantIdKey,
      how="left",
      suffixes=("", "_note_author"),
      unsafeAllowed={c.postSelectionValueKey, c.postSelectionValueKey + "_note_author"},
    )
  )
  ratingsWithNoPostSelectionSimilarityValue = ratingsWithPostSelectionSimilarity[
    pd.isna(ratingsWithPostSelectionSimilarity[c.postSelectionValueKey])
  ]
  ratingsWithPostSelectionSimilarityValue = ratingsWithPostSelectionSimilarity[
    (~pd.isna(ratingsWithPostSelectionSimilarity[c.postSelectionValueKey]))
    & (
      ratingsWithPostSelectionSimilarity[c.postSelectionValueKey]
      != ratingsWithPostSelectionSimilarity[c.postSelectionValueKey + "_note_author"]
    )
  ]
  ratingsWithPostSelectionSimilarityValue.sort_values(
    by=[c.noteIdKey, c.createdAtMillisKey], ascending=True, inplace=True
  )
  ratingsWithPostSelectionSimilarityValue.drop_duplicates(
    subset=[c.noteIdKey, c.postSelectionValueKey], keep="first", inplace=True
  )

  if len(notes) < c.minNumNotesForProdData:
    return ratings

  ratings = pd.concat(
    [ratingsWithPostSelectionSimilarityValue, ratingsWithNoPostSelectionSimilarityValue], axis=0
  )
  ratings.drop(
    columns={c.noteAuthorParticipantIdKey, c.raterParticipantIdKey + "_note_author"},
    errors="ignore",
    inplace=True,
  )
  logger.info(f"Total ratings after to applying post selection similarity: {len(ratings)}")
  logger.info(
    f"Total unique ratings after: {len(ratings[[c.noteIdKey, c.raterParticipantIdKey]].drop_duplicates())}"
  )
  return ratings


def _preprocess_ratings(notes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
  """
  Preprocess ratings dataframe.
  """
  ratings = notes[[c.noteIdKey, c.tweetIdKey]].merge(
    ratings[[c.raterParticipantIdKey, c.noteIdKey, c.createdAtMillisKey]],
    on=c.noteIdKey,
    how="inner",
  )
  ratings = ratings[(ratings[c.tweetIdKey] != -1) & (ratings[c.tweetIdKey] != "-1")]
  return ratings


def _min_survivable_co_rating_count(
  N: int,
  pmiPseudocounts: int,
  minSimPseudocounts: int,
  smoothedNpmiThreshold: float,
  minimumRatingProportionThreshold: float,
  minUniquePosts: int,
) -> int:
  """Minimum co-rating count that could pass any condition in the filter below.

  Used as a pre-filter to avoid materializing billions of pairs that would be
  discarded anyway.  Must be kept in sync with the filter conditions in
  _join_rater_totals_compute_pmi_and_filter_edges_below_threshold.
  """
  # minSim condition: coRatings / (minTotal + minSimPseudocounts) >= threshold
  # Most favorable case: minTotal = minUniquePosts (smallest allowed)
  min_sim_min = int(minimumRatingProportionThreshold * (minUniquePosts + minSimPseudocounts))

  # NPMI condition: smoothedNpmi >= threshold
  # Most favorable case: both rater totals = minUniquePosts (smallest denominator)
  npmi_min = min_sim_min  # fallback: NPMI never passes below minSim minimum
  denom = (minUniquePosts + pmiPseudocounts) ** 2
  for count in range(1, min_sim_min):
    pmi = np.log(count * N / denom)
    npmi = pmi / -np.log(count / N)
    if npmi >= smoothedNpmiThreshold:
      npmi_min = count
      break

  return min(min_sim_min, npmi_min)


def _join_rater_totals_compute_pmi_and_filter_edges_below_threshold(
  pairCountsDict: Dict,
  raterTotalsDict: Dict,
  N: int,
  pmiPseudocounts: int,
  minSimPseudocounts: int,
  smoothedNpmiThreshold: float,
  minimumRatingProportionThreshold: float,
):
  keys_to_delete = []

  with c.time_block("Compute PMI and minSim"):
    for leftRaterId, rightRaterId in pairCountsDict:
      if leftRaterId not in raterTotalsDict or rightRaterId not in raterTotalsDict:
        keys_to_delete.append((leftRaterId, rightRaterId))
        continue

      leftTotal = raterTotalsDict[leftRaterId]
      rightTotal = raterTotalsDict[rightRaterId]
      coRatings = pairCountsDict[(leftRaterId, rightRaterId)]

      if type(coRatings) != int:
        # already processed (should only occur when re-running...)
        continue

      # PMI
      pmiNumerator = coRatings * N
      pmiDenominator = (leftTotal + pmiPseudocounts) * (rightTotal + pmiPseudocounts)
      smoothedPmi = np.log(pmiNumerator / pmiDenominator)
      smoothedNpmi = smoothedPmi / -np.log(coRatings / N)

      # minSim
      minTotal = min(leftTotal, rightTotal)
      minSimRatingProp = coRatings / (minTotal + minSimPseudocounts)

      if (smoothedNpmi >= smoothedNpmiThreshold) or (
        minSimRatingProp >= minimumRatingProportionThreshold
      ):
        pairCountsDict[(leftRaterId, rightRaterId)] = (smoothedNpmi, minSimRatingProp)
      else:
        keys_to_delete.append((leftRaterId, rightRaterId))

    print(f"Pairs dict used {sys.getsizeof(pairCountsDict) * 1e-9}GB RAM at max")

  with c.time_block("Delete unneeded pairs from pairCountsDict"):
    for key in keys_to_delete:
      del pairCountsDict[key]

  print(
    f"Pairs dict used {sys.getsizeof(pairCountsDict) * 1e-9}GB RAM after deleted unneeded pairs"
  )

  return pairCountsDict


def aggregate_into_cliques(suspectPairs):
  with c.time_block("Aggregate into cliques by post selection similarity"):
    userToCliqueMap = dict()
    cliqueToUserMap = dict()

    nextNewCliqueId = 1  # start cliqueIdxs from 1

    for sid, tid in suspectPairs:
      if sid in userToCliqueMap:
        if tid in userToCliqueMap:
          # both in map. merge if not same clique
          if userToCliqueMap[sid] != userToCliqueMap[tid]:
            # merge. assign all member's of target clique to source clique.
            # slow way: iterate over all values here.
            # fast way: maintain a reverse map of cliqueToUserMap.
            sourceDestClique = userToCliqueMap[sid]
            oldTargetCliqueToDel = userToCliqueMap[tid]

            for userId in cliqueToUserMap[oldTargetCliqueToDel]:
              cliqueToUserMap[sourceDestClique].append(userId)
              userToCliqueMap[userId] = sourceDestClique
            del cliqueToUserMap[oldTargetCliqueToDel]
            gc.collect()

        else:
          # source in map; target not. add target to source's clique
          sourceClique = userToCliqueMap[sid]
          userToCliqueMap[tid] = sourceClique
          cliqueToUserMap[sourceClique].append(tid)
      elif tid in userToCliqueMap:
        # target in map; source not. add source to target's clique
        targetClique = userToCliqueMap[tid]
        userToCliqueMap[sid] = targetClique
        cliqueToUserMap[targetClique].append(sid)
      else:
        # new clique
        userToCliqueMap[sid] = nextNewCliqueId
        userToCliqueMap[tid] = nextNewCliqueId
        cliqueToUserMap[nextNewCliqueId] = [sid, tid]
        nextNewCliqueId += 1
  return cliqueToUserMap, userToCliqueMap


@njit(cache=True)
def _count_pair_events(times, raters, group_starts, n_groups, n_total, window_millis):
  """Count total number of times a pair of raters co-rated a note within a sliding time window."""
  total = int64(0)
  for g in range(n_groups):
    start = group_starts[g]
    end = group_starts[g + 1] if g + 1 < n_groups else n_total
    if end - start < 2:
      continue
    window_start = start
    for i in range(start, end):
      while times[window_start] < times[i] - window_millis:
        window_start += 1
      for j in range(window_start, i):
        if raters[i] != raters[j]:
          total += 1
  return total


@njit(cache=True)
def _fill_pair_events(times, raters, group_starts, group_tweet_ids,
                      n_groups, n_total, window_millis,
                      out_tweets, out_left, out_right):
  """Fill pre-allocated arrays with (tweet, rater_l, rater_r) pair events."""
  pos = int64(0)
  for g in range(n_groups):
    start = group_starts[g]
    end = group_starts[g + 1] if g + 1 < n_groups else n_total
    if end - start < 2:
      continue
    tweet = group_tweet_ids[g]
    window_start = start
    for i in range(start, end):
      while times[window_start] < times[i] - window_millis:
        window_start += 1
      for j in range(window_start, i):
        if raters[i] == raters[j]:
          continue
        out_tweets[pos] = tweet
        if raters[i] < raters[j]:
          out_left[pos] = raters[i]
          out_right[pos] = raters[j]
        else:
          out_left[pos] = raters[j]
          out_right[pos] = raters[i]
        pos += 1
  return pos


def _log_mem(label):
  avail = _mem_available_gb()
  avail_str = f", available: {avail:.1f}GB" if avail is not None else ""
  logger.info(f"[mem] {label}: RSS {_mem_gb():.1f}GB{avail_str}")


def _get_pair_counts_dict(ratings, windowMillis, minCoRatingCount=5):
  """
  Returns a dict of (rater_l, rater_r) => count of how many times they co-rated distinct tweets within windowMillis.
  
  Pairs with count < minCoRatingCount are filtered out (they can't survive the downstream PMI/minSim filter anyway).
  """
  logger.info("Starting pair counts computation")
  _log_mem("start")

  # Factorize rater/tweet IDs to int32 codes
  rater_codes, rater_uniques = pd.factorize(ratings[c.raterParticipantIdKey])
  rater_arr = rater_codes.astype(np.int32)
  n_raters = len(rater_uniques)
  del rater_codes

  # Factorize tweet IDs to int32 codes
  tweet_codes, tweet_uniques = pd.factorize(ratings[c.tweetIdKey])
  tweet_arr = tweet_codes.astype(np.int32)
  n_tweets = len(tweet_uniques)
  del tweet_codes, tweet_uniques

  # Factorize note IDs to int32 codes
  note_codes, _ = pd.factorize(ratings[c.noteIdKey])
  note_arr = note_codes.astype(np.int32)
  del note_codes

  # Extract times as numpy int64 array
  times = ratings[c.createdAtMillisKey].values.astype(np.int64)
  _log_mem("after extracting arrays")

  # Sort by (note, time) for contiguous grouped access
  sort_idx = np.lexsort((times, note_arr))
  tweet_arr = tweet_arr[sort_idx]
  note_arr = note_arr[sort_idx]
  times = times[sort_idx]
  rater_arr = rater_arr[sort_idx]
  del sort_idx
  gc.collect()

  # Compute note-group boundaries
  n = len(tweet_arr)
  group_mask = np.empty(n, dtype=np.bool_)
  group_mask[0] = True
  group_mask[1:] = note_arr[1:] != note_arr[:-1]
  group_starts = np.where(group_mask)[0].astype(np.int64)
  n_groups = len(group_starts)

  # Extract tweet IDs for each note group
  group_tweet_ids = tweet_arr[group_starts]
  del tweet_arr, note_arr, group_mask
  gc.collect()
  _log_mem(f"after preprocessing: {n:,} ratings, {n_groups:,} groups")

  # Count total number of times a pair of raters co-rated a note within windowMillis
  logger.info("Counting pair events...")
  total_events = _count_pair_events(
    times, rater_arr, group_starts, int64(n_groups), int64(n), int64(windowMillis),
  )
  logger.info(f"Total pair events: {total_events:,}")
  _log_mem("after count pass")

  if total_events == 0:
    return {}

  # --- Pass 2: fill arrays ---
  logger.info("Generating pair events...")
  out_tweets = np.empty(total_events, dtype=np.int32)
  out_left = np.empty(total_events, dtype=np.int32)
  out_right = np.empty(total_events, dtype=np.int32)
  _log_mem("after allocating output arrays")

  # this will fill the out_tweets, out_left, out_right arrays with the (tweet, rater_l, rater_r) pair events
  _fill_pair_events(
    times, rater_arr, group_starts, group_tweet_ids,
    int64(n_groups), int64(n), int64(windowMillis),
    out_tweets, out_left, out_right,
  )

  del times, rater_arr, group_starts, group_tweet_ids
  gc.collect()
  _log_mem("after fill pass")

  # Pack the (rater_l, rater_r, tweet) triple into a single int64 using bit-shifting
  rater_bits = max(1, int(np.ceil(np.log2(max(n_raters, 2)))))
  tweet_bits = max(1, int(np.ceil(np.log2(max(n_tweets, 2)))))
  assert 2 * rater_bits + tweet_bits <= 63, (
    f"IDs too large for int63 packing: {rater_bits}*2 + {tweet_bits} = {2 * rater_bits + tweet_bits}"
  )
  left_shift = rater_bits + tweet_bits
  right_shift = tweet_bits

  compound = (
    (out_left.astype(np.int64) << left_shift) |
    (out_right.astype(np.int64) << right_shift) |
    out_tweets.astype(np.int64)
  )
  del out_left, out_right, out_tweets
  gc.collect()

  # Sort the packed pair events so that we can deduplicate them
  logger.info("Sorting pair events...")
  compound.sort()
  _log_mem("after sort")

  # Dedup: consecutive equal values = same (rater_l, rater_r, tweet) from multiple notes
  dup_mask = np.empty(len(compound), dtype=np.bool_)
  dup_mask[0] = True
  dup_mask[1:] = compound[1:] != compound[:-1]
  compound = compound[dup_mask]
  del dup_mask
  gc.collect()
  logger.info(f"After per-tweet dedup: {len(compound):,} unique pair-tweet events")
  _log_mem("after dedup")

  # Count per pair: drop tweet bits to get pair key, then run-length encode
  pair_key = compound >> tweet_bits
  pair_boundary = np.empty(len(pair_key), dtype=np.bool_)
  pair_boundary[0] = True
  pair_boundary[1:] = pair_key[1:] != pair_key[:-1]
  pair_starts = np.where(pair_boundary)[0]
  pair_counts_arr = np.diff(np.append(pair_starts, len(pair_key)))
  del pair_key, pair_boundary
  gc.collect()
  logger.info(f"Total unique pairs: {len(pair_starts):,}")

  # Filter by minimum count, extract rater codes from surviving compound keys
  count_mask = pair_counts_arr >= minCoRatingCount
  surviving = compound[pair_starts[count_mask]]
  surviving_counts = pair_counts_arr[count_mask]
  del compound, pair_starts, pair_counts_arr, count_mask
  gc.collect()

  rater_mask = (1 << rater_bits) - 1
  result_left = ((surviving >> left_shift) & rater_mask).astype(np.int32)
  result_right = ((surviving >> right_shift) & rater_mask).astype(np.int32)
  logger.info(f"Pairs with count >= {minCoRatingCount}: {len(result_left):,}")
  _log_mem("after filtering")

  # Convert to dict with original rater IDs
  pair_counts = {}
  for i in range(len(result_left)):
    ri = rater_uniques[result_left[i]]
    rj = rater_uniques[result_right[i]]
    pair_counts[tuple(sorted((ri, rj)))] = int(surviving_counts[i])

  logger.info(f"Output: {len(pair_counts):,} pairs")
  _log_mem("done")
  return pair_counts
