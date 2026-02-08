import gc
import logging
import sys
from typing import Dict

from . import constants as c, dev_cache

import numpy as np
import pandas as pd


logger = logging.getLogger("birdwatch.post_selection_similarity")
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
    # this takes forever to run
    with c.time_block("Compute pair counts dict"):
      self.pairCountsDict = _get_pair_counts_dict(self.ratings, windowMillis=windowMillis)
    logger.info(f"Computed pair counts dict for {len(self.pairCountsDict)} pairs")

    self.uniqueRatingsOnTweets = self.ratings[
      [c.tweetIdKey, c.raterParticipantIdKey]
    ].drop_duplicates()
    logger.info(f"Computed unique ratings on tweets for {len(self.uniqueRatingsOnTweets)} ratings")
    raterTotals = self.uniqueRatingsOnTweets[c.raterParticipantIdKey].value_counts()
    raterTotalsDict = {
      index: value for index, value in raterTotals.items() if value >= minUniquePosts
    }

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
    Returns dataframe with [raterParticipantId, postSelectionSimilarityValue] columns.
    postSelectionSimilarityValue is None by default.
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


def _get_pair_counts_dict(ratings, windowMillis):
  pair_counts = dict()

  # Group by tweetIdKey to process each tweet individually
  grouped_by_tweet = ratings.groupby(c.tweetIdKey, sort=False)

  for _, tweet_group in grouped_by_tweet:
    # Keep track of pairs we've already counted for this tweetId
    pairs_counted_in_tweet = set()

    # Group by noteIdKey within the tweet
    grouped_by_note = tweet_group.groupby(c.noteIdKey, sort=False)

    for _, note_group in grouped_by_note:
      note_group.sort_values(c.createdAtMillisKey, inplace=True)

      # Extract relevant columns as numpy arrays for efficient computation
      times = note_group[c.createdAtMillisKey].values
      raters = note_group[c.raterParticipantIdKey].values

      n = len(note_group)
      window_start = 0

      for i in range(n):
        # Move the window start forward if the time difference exceeds windowMillis
        while times[i] - times[window_start] > windowMillis:
          window_start += 1

        # For all indices within the sliding window (excluding the current index)
        for j in range(window_start, i):
          if raters[i] != raters[j]:
            left_rater, right_rater = tuple(sorted((raters[i], raters[j])))
            pair = (left_rater, right_rater)
            # Only count this pair once per tweetId
            if pair not in pairs_counted_in_tweet:
              pairs_counted_in_tweet.add(pair)
              # Update the count for this pair
              if pair not in pair_counts:
                pair_counts[pair] = 0
              pair_counts[pair] += 1

  return pair_counts
