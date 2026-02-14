"""Assign notes to a set of predetermined topics.

The topic assignment process is seeded with a small set of terms which are indicative of
the topic.  After preliminary topic assignment based on term matching, a logistic regression
trained on bag-of-words features model expands the set of in-topic notes for each topic.
Note that the logistic regression modeling excludes any tokens containing seed terms.

This approach represents a prelimiary approach to topic assignment while Community Notes
evaluates the efficacy of per-topic note scoring.
"""

from itertools import product
import logging
import re
from typing import List, Optional, Tuple

from . import constants as c
from .enums import Topics

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid, softmax
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline


logger = logging.getLogger("birdwatch.topic_model")
logger.setLevel(logging.INFO)


seedTerms = {
  Topics.UkraineConflict: {
    "ukrain",  # intentionally shortened for expanded matching
    "russia",
    "kiev",
    "kyiv",
    "moscow",
    "zelensky",
    "putin",
  },
  Topics.GazaConflict: {
    "israel",
    "palestin",  # intentionally shortened for expanded matching
    "gaza",
    "jerusalem",
    "\shamas\s",
  },
  Topics.MessiRonaldo: {
    "messi\s",  # intentional whitespace to prevent prefix matches
    "ronaldo",
  },
  Topics.Scams: {
    "scam",
    "undisclosed\sad",  # intentional whitespace
    "terms\sof\sservice",  # intentional whitespace
    "help\.x\.com",
    "x\.com/tos",
    "engagement\sfarm",  # intentional whitespace
    "spam",
    "gambling",
    "apostas",
    "apuestas",
    "dropship",
    "drop\sship",  # intentional whitespace
    "promotion",
  },
}


def get_seed_term_with_periods():
  seedTermsWithPeriods = []
  for terms in seedTerms.values():
    for term in terms:
      if "\." in term:
        seedTermsWithPeriods.append(term)
  return seedTermsWithPeriods


class TopicModel(object):
  def __init__(self, unassignedThreshold=0.99):
    """Initialize a list of seed terms for each topic."""
    self._seedTerms = seedTerms
    self._unassignedThreshold = unassignedThreshold
    # Pre-build tokenizer components once (reused per-text by custom_tokenizer)
    self._preprocessor = CountVectorizer(
      strip_accents="unicode", lowercase=True
    ).build_preprocessor()
    seed_pats = [
      r"(?:https?://)?(" + term + r")(?:/[^\s]+)?|" for term in get_seed_term_with_periods()
    ]
    self._token_pattern = re.compile(r"(?i)" + "".join(seed_pats + [r"\b\w\w+\b"]))
    # Cache for custom_tokenizer results — the tokenizer is called twice on the same
    # texts (once in _get_stop_words and once in the pipeline's CountVectorizer.fit),
    # so the second pass hits the cache and is essentially free.
    self._tokenizer_cache = {}

  def _make_seed_labels(self, texts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Produce a label vector based on seed terms.

    Eliminates regex entirely by converting all seed patterns to plain string
    searches using Python's ``in`` operator (C-level Boyer-Moore).

    Texts are lowercased, whitespace-normalized to spaces, and prepended with
    a space so that searching for ``" term"`` handles both the ``(\\s|^)``
    boundary conditions in one plain string check.  Patterns containing ``\\.``
    (URLs) are searched without a boundary prefix.

    Args:
      texts: array containing strings for topic assignment

    Returns:
      Tuple[0]: array specifying topic labels for texts
      Tuple[1]: array specifying texts that are unassigned due to conflicting matches.
    """
    n = len(texts)
    labels = np.zeros(n, dtype=np.int64)
    conflictedTexts = np.zeros(n, dtype=bool)
    if n == 0:
      return labels, conflictedTexts

    # Normalize: lowercase, replace all whitespace with spaces, prepend a space.
    # The prepended space lets us use " term" as a search pattern to handle both
    # (\s|^) boundary conditions (term after whitespace, or term at start of text).
    _WS_TRANS = str.maketrans("\t\n\r\x0b\x0c", "     ")
    texts_norm = pd.Series([
      " " + t.lower().translate(_WS_TRANS) if isinstance(t, str) else " "
      for t in texts
    ])

    # Convert each seed term to a plain search string:
    # - Patterns with \.: strip the escapes (e.g. help\.x\.com -> help.x.com), no boundary
    # - All others: replace \s with space, prepend " " for boundary
    topic_masks = []
    topic_values = []
    for topic, patterns in self._seedTerms.items():
      topic_mask = np.zeros(n, dtype=bool)
      for pattern in patterns:
        if "\\." in pattern:
          search_term = pattern.replace("\\.", ".")
        else:
          search_term = " " + pattern.replace("\\s", " ")
        topic_mask |= texts_norm.str.contains(search_term, regex=False, na=False).values
      topic_masks.append(topic_mask)
      topic_values.append(topic.value)

    # Stack into (n_texts x n_topics) matrix and count matches per text
    match_matrix = np.column_stack(topic_masks)
    topic_values_arr = np.array(topic_values, dtype=np.int64)
    match_count = match_matrix.sum(axis=1)

    # Single-topic match: assign that topic
    single = match_count == 1
    if single.any():
      labels[single] = topic_values_arr[np.argmax(match_matrix[single], axis=1)]

    # Multi-topic match: conflicted
    multi = match_count > 1
    labels[multi] = Topics.Unassigned.value
    conflictedTexts[multi] = True

    unassigned_count = np.sum(conflictedTexts)
    logger.info(f"  Notes unassigned due to multiple matches: {unassigned_count}")
    return labels, conflictedTexts

  def custom_tokenizer(self, text):
    """Tokenize text, capturing seed-term URLs as single tokens.

    Uses pre-compiled preprocessor and regex pattern from __init__, and caches
    results so that the second tokenization pass (pipeline fit after stop-word
    identification) is essentially free.
    """
    cached = self._tokenizer_cache.get(text)
    if cached is not None:
      return cached

    processed = self._preprocessor(text)
    tokens = []
    for match in self._token_pattern.finditer(processed):
      seed_term = next((g for g in match.groups() if g is not None), None)
      tokens.append(seed_term if seed_term is not None else match.group(0))

    self._tokenizer_cache[text] = tokens
    return tokens

  def _get_stop_words(self, texts: np.ndarray) -> List[str]:
    """Identify tokens in the extracted vocabulary that contain seed terms.

    Any token containing a seed term will be treated as a stop word (i.e. removed
    from the extracted features).  This prevents the model from training on the same
    tokens used to label the data.

    Args:
      texts: array containing strings for topic assignment

    Returns:
      List specifying which tokens to exclude from the features.
    """
    # Extract vocabulary
    cv = CountVectorizer(tokenizer=self.custom_tokenizer, token_pattern=None)
    cv.fit(texts)
    rawVocabulary = cv.vocabulary_.keys()
    logger.info(f"  Initial vocabulary length: {len(rawVocabulary)}")
    # Identify stop words
    blockedTokens = set()
    for terms in self._seedTerms.values():
      # Remove whitespace and any escaped whitespace characters from seed terms
      blockedTokens |= {re.sub(r"\\s", "", t.strip()) for t in terms}
      # Convert escaped periods to periods
      blockedTokens |= {re.sub(r"\\.", ".", t.strip()) for t in terms}
    logger.info(f"  Total tokens to filter: {len(blockedTokens)}")
    stopWords = [v for v in rawVocabulary if any(t in v for t in blockedTokens)]
    logger.info(f"  Total identified stopwords: {len(stopWords)}")
    return stopWords

  def _merge_predictions_and_labels(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Update predictions based on defined labels when the label is not Unassigned.

    Args:
      probs: 2D matrix specifying the likelihood of each class

    Returns:
      Updated predictions based on keyword matches when available.
    """
    predictions = np.argmax(probs, axis=1)
    for label in range(1, len(Topics)):
      # Update label if (1) note was assigned based on the labeling heuristic, and (2)
      # p(Unassigned) is below the required uncertainty threshold.
      predictions[(labels == label) & (probs[:, 0] <= self._unassignedThreshold)] = label
    return predictions

  def _prepare_post_text(self, notes: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all notes within each post into a single row associated with the post.

    Args:
      notes: dataframe containing raw note text

    Returns:
      DataFrame with one post per row containing note text
    """
    postNoteText = (
      notes[[c.tweetIdKey, c.summaryKey]]
      .fillna({c.summaryKey: ""})
      .groupby(c.tweetIdKey)[c.summaryKey]
      .apply(lambda postNotes: " ".join(postNotes))
      .reset_index(drop=False)
    )
    # Default tokenization for CountVectorizer will not split on underscore, which
    # results in very long tokens containing many words inside of URLs.  Removing
    # underscores allows us to keep default splitting while fixing that problem.
    postNoteText[c.summaryKey] = [
      text.replace("_", " ") for text in postNoteText[c.summaryKey].values
    ]
    return postNoteText

  def train_individual_note_topic_classifier(
    self, postText: pd.DataFrame
  ) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    with c.time_block("Get Note Topics: Make Seed Labels"):
      seedLabels, conflictedTexts = self._make_seed_labels(postText[c.summaryKey].values)

    with c.time_block("Get Note Topics: Get Stop Words"):
      stopWords = self._get_stop_words(postText[c.summaryKey].values)

    with c.time_block("Get Note Topics: Train Model"):
      # Define and fit model
      pipe = Pipeline(
        [
          (
            "UnigramEncoder",
            CountVectorizer(
              tokenizer=self.custom_tokenizer,
              token_pattern=None,
              strip_accents="unicode",
              stop_words=stopWords,
              min_df=25,
              max_df=max(1000, int(0.25 * len(postText))),
            ),
          ),
          ("tfidf", TfidfTransformer()),
          ("Classifier", LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1)),
        ],
        verbose=True,
      )
      pipe.fit(
        # Notice that we omit posts with an unclear label from training.
        postText[c.summaryKey].values[~conflictedTexts],
        seedLabels[~conflictedTexts],
      )
    return pipe, seedLabels, conflictedTexts

  def train_note_topic_classifier(
    self, notes: pd.DataFrame
  ) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    # Obtain aggregate post text, seed labels and stop words
    with c.time_block("Get Note Topics: Prepare Post Text"):
      postText = self._prepare_post_text(notes)
    pipe, seedLabels, conflictedTexts = self.train_individual_note_topic_classifier(postText)
    return pipe, seedLabels, conflictedTexts

  def train_bootstrapped_note_topic_classifier(
    self,
    notes: pd.DataFrame,
  ) -> Tuple[List[Pipeline], List[np.ndarray], List[np.ndarray]]:
    # Obtain aggregate post text, seed labels and stop words
    pipes = []
    seedLabelSets = []
    conflictedTextSetsForAccuracyEval = []
    with c.time_block("Get Note Topics: Prepare Post Text"):
      postText = self._prepare_post_text(notes)
    pipe, seedLabels, conflictedTextsForAccuracyEval = self.train_individual_note_topic_classifier(
      postText
    )
    pipes.append(pipe)
    seedLabelSets.append(seedLabels)
    conflictedTextSetsForAccuracyEval.append(conflictedTextsForAccuracyEval)
    # train and append additional ablated seed word sets for all combos for topics 1 and 2,
    # plus 3 and 4 individually
    gazaUkrCombinations = list(
      product(list(seedTerms[Topics.UkraineConflict]), list(seedTerms[Topics.GazaConflict]))
    )
    for i in range(len(gazaUkrCombinations)):
      bootstrappedSeedTerms = {}
      bootstrappedSeedTerms[Topics.UkraineConflict] = seedTerms[Topics.UkraineConflict].copy()
      bootstrappedSeedTerms[Topics.UkraineConflict].remove(gazaUkrCombinations[i][0])
      bootstrappedSeedTerms[Topics.GazaConflict] = seedTerms[Topics.GazaConflict].copy()
      bootstrappedSeedTerms[Topics.GazaConflict].remove(gazaUkrCombinations[i][1])
      bootstrappedSeedTerms[Topics.MessiRonaldo] = seedTerms[Topics.MessiRonaldo].copy()
      bootstrappedSeedTerms[Topics.MessiRonaldo].remove(
        np.random.choice(list(seedTerms[Topics.MessiRonaldo]), 1)[0]
      )
      bootstrappedSeedTerms[Topics.Scams] = seedTerms[Topics.Scams].copy()
      bootstrappedSeedTerms[Topics.Scams].remove(
        np.random.choice(list(seedTerms[Topics.Scams]), 1)[0]
      )
      self._seedTerms = bootstrappedSeedTerms
      (
        pipe,
        seedLabels,
        conflictedTextsForAccuracyEval,
      ) = self.train_individual_note_topic_classifier(postText)
      pipes.append(pipe)
      seedLabelSets.append(seedLabels)
      conflictedTextSetsForAccuracyEval.append(conflictedTextsForAccuracyEval)
    self._seedTerms = seedTerms
    return pipes, seedLabelSets, conflictedTextSetsForAccuracyEval

  def get_note_topics(
    self,
    notes: pd.DataFrame,
    noteTopicClassifiers: Optional[List[Pipeline]] = None,
    seedLabelSets: Optional[List[np.ndarray]] = None,
    conflictedTextSetsForAccuracyEval: Optional[List[np.ndarray]] = None,
    bootstrapped: Optional[bool] = False,
    assignConflicted: Optional[bool] = False,
    exitOnLowAccuracy: Optional[bool] = True,
  ) -> pd.DataFrame:
    """Return a DataFrame specifying each {note, topic} pair.

    Notes that are not assigned to a topic do not appear in the dataframe.

    Args:
      notes: DF containing all notes to potentially assign to a topic
    """
    logger.info("Assigning notes to topics:")
    if noteTopicClassifiers is not None:
      pipes = noteTopicClassifiers
    else:
      logger.info("Training note topic classifier")
      if bootstrapped:
        (
          pipes,
          seedLabelSets,
          conflictedTextSetsForAccuracyEval,
        ) = self.train_bootstrapped_note_topic_classifier(notes)
      else:
        (
          pipe,
          seedLabelSet,
          conflictedTextForAccuracyEval,
        ) = self.train_note_topic_classifier(notes)
        pipes, seedLabelSets, conflictedTextSetsForAccuracyEval = (
          [pipe],
          [seedLabelSet],
          [conflictedTextForAccuracyEval],
        )
    with c.time_block("Get Note Topics: Prepare Post Text"):
      postText = self._prepare_post_text(notes)

    labelSets = []
    if seedLabelSets is None:
      seedLabelSets = [None for i in range(len(pipes))]
    if conflictedTextSetsForAccuracyEval is None:
      conflictedTextSetsForAccuracyEval = [None for i in range(len(pipes))]

    with c.time_block("Get Note Topics: Predict"):
      # Predict notes.  Notice that in effect we are looking to see which notes in the
      # training data the model felt were mis-labeled after the training process
      # completed, and generating labels for any posts which were omitted from the
      # original training.
      for i, pipe in enumerate(pipes):
        assert type(pipe) == Pipeline, "unsupported classifier"
        logits = pipe.decision_function(postText[c.summaryKey].values)
        # Transform logits to probabilities, handling the case where logits are 1D because
        # of unit testing with only 2 topics.
        if len(logits.shape) == 1:
          probs = sigmoid(logits)
          probs = np.vstack([1 - probs, probs]).T
        else:
          probs = softmax(logits, axis=1)

        if seedLabelSets[i] is None:
          with c.time_block("Get Note Topics: Make Seed Labels"):
            seedLabelSets[i], _ = self._make_seed_labels(postText[c.summaryKey].values)

        if conflictedTextSetsForAccuracyEval[i] is not None:
          self.validate_note_topic_accuracy_on_seed_labels(
            np.argmax(probs, axis=1),
            seedLabelSets[i],
            conflictedTextSetsForAccuracyEval[i],
            exitOnLowAccuracy,
          )

        with c.time_block("Get Note Topics: Merge and assign predictions"):
          topicAssignments = self._merge_predictions_and_labels(probs, seedLabelSets[i])
          logger.info(f"  Post Topic assignment results: {np.bincount(topicAssignments)}")

          # Assign topics to notes based on aggregated note text, and drop any
          # notes on posts that were unassigned.
          postTextCopy = postText.copy()
          postTextCopy[c.noteTopicKey] = [Topics(t).name for t in topicAssignments]
          postTextCopy = postTextCopy[postTextCopy[c.noteTopicKey] != Topics.Unassigned.name]
          noteTopics = notes[[c.noteIdKey, c.tweetIdKey]].merge(
            postTextCopy[[c.tweetIdKey, c.noteTopicKey]]
          )
          print(noteTopics.shape)
          labelSets.append(noteTopics.drop(columns=c.tweetIdKey))
        logger.info(
          f"  Note Topic assignment results:\n{noteTopics[c.noteTopicKey].value_counts(dropna=False)}"
        )
    if len(labelSets) == 1:
      return noteTopics.drop(columns=c.tweetIdKey)
    else:
      noteTopics = pd.concat(labelSets)
      noteTopics["cnt"] = 1
      numTopics = (
        noteTopics[[c.noteIdKey, c.noteTopicKey]]
        .drop_duplicates()
        .groupby([c.noteIdKey])
        .agg("count")
        .reset_index()
      )
      conflicted = numTopics.loc[numTopics[c.noteTopicKey] > 1]
      if assignConflicted == True:
        # assign to most common result
        return (
          noteTopics.groupby([c.noteIdKey, c.noteTopicKey])
          .agg("count")
          .reset_index()
          .sort_values("cnt", ascending=False)
          .groupby(c.noteIdKey)
          .head(1)
          .drop(columns="cnt")
        )
      else:
        return (
          noteTopics.loc[~(noteTopics[c.noteIdKey].isin(conflicted[c.noteIdKey].values))]
          .groupby([c.noteIdKey, c.noteTopicKey])
          .head(1)
          .drop(columns="cnt")
        )

  def validate_note_topic_accuracy_on_seed_labels(
    self, pred, seedLabels, conflictedTexts, exitOnLowAccuracy=True
  ):
    balancedAccuracy = balanced_accuracy_score(seedLabels[~conflictedTexts], pred[~conflictedTexts])
    logger.info(f"  Balanced accuracy on raw predictions: {balancedAccuracy}")
    if exitOnLowAccuracy:
      assert balancedAccuracy > 0.35, f"Balanced accuracy too low: {balancedAccuracy}"
    # Validate that any conflicted text is Unassigned in seedLabels
    assert all(seedLabels[conflictedTexts] == Topics.Unassigned.value)
