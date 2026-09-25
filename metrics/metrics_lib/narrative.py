"""
Narrative diversity (D_narr): sentiment-arc divergence across a group of texts

Per text: split into ARC_LENGTH equal-proportion narrative segments, average
per-sentence sentiment within each segment -> one sentiment arc vector per text

D_narr for a group = mean pairwise Euclidean distance between arc vectors (Reagan et al., 2016)

    from metrics_lib.narrative import narrative_diversity
    d_narr = narrative_diversity(list_of_texts_for_one_prompt)

Two sentiment scorers are available via `method`:
  - "transformer" (default): siebert/sentiment-roberta-large-english.
    needs a GPU/CPU capable of running a ~1.4GB transformer model.
  - "vader": vaderSentiment (lexicon-based). Much faster, no model download;
    kept in the original notebook as a reference/validation method.
"""
from functools import lru_cache
from itertools import combinations
import numpy as np


# Number of narrative segments to split each text into
ARC_LENGTH = 10

# Must be >= ARC_LENGTH so every bin gets >= 1 sentence
MIN_SENTENCES = 10


@lru_cache(maxsize=1)
def _ensure_punkt():
    """Download NLTK's Punkt tokenizer data only if not already present, and only once per run."""
    import nltk

    for resource in ('tokenizers/punkt', 'tokenizers/punkt_tab'):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1], quiet=True)

    return True


def _sent_tokenize(text):
    """Tokenize text into sentences using NLTK's Punkt tokenizer"""
    _ensure_punkt()

    from nltk.tokenize import sent_tokenize

    return sent_tokenize(text)


def bin_to_arc(scores, arc_length: int = ARC_LENGTH):
    """Split per-sentence scores into `arc_length` contiguous segments and average each."""

    # Split into `arc_length` contiguous segments of near-equal size.
    # The last segment may be shorter if len(scores) is not divisible by arc_length.
    bins = np.array_split(np.asarray(scores), arc_length)

    return np.array([b.mean() for b in bins])


# Use cached transformer pipeline to avoid repeated model loading
@lru_cache(maxsize=1)
def _get_transformer_pipeline():
    from transformers import pipeline
    import torch

    # Prefer an accelerator: CUDA (Linux/Colab), then MPS (Apple Silicon), else CPU.
    # MPS was disabled here historically but is stable for this model on torch >= 2.1.
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = -1

    # Use the "siebert/sentiment-roberta-large-english" model
    return pipeline(
        'sentiment-analysis',
        model='siebert/sentiment-roberta-large-english',
        device=device,
        truncation=True,
    )


# Use cached VADER analyzer to avoid repeated model loading
@lru_cache(maxsize=1)
def _get_vader_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def get_sentiment_arc(text: str, method: str = 'transformer', arc_length: int = ARC_LENGTH,
                       min_sentences: int = MIN_SENTENCES, batch_size: int = 32):
    """Sentiment arc for one text or None if it has fewer than min_sentences sentences."""

    # Split text into sentences (keep every non-empty sentence, no length / word-count filter)
    sentences = [s.strip() for s in _sent_tokenize(text) if s.strip()]

    # Filter out texts that are too short to produce a full arc
    if len(sentences) < min_sentences:
        return None

    # Compute per-sentence sentiment score via Siebert
    if method == 'transformer':
        pipe = _get_transformer_pipeline()
        results = pipe(sentences, batch_size=batch_size)

        # Continuous signed score in [-1, 1]: 2*P(positive) - 1, so a 50/50-uncertain sentence scores around 0
        p_positive = np.array([r['score'] if r['label'] == 'POSITIVE' else 1 - r['score'] for r in results])
        scores = 2 * p_positive - 1

    # Compute per-sentence sentiment score via VADER
    elif method == 'vader':
        analyzer = _get_vader_analyzer()
        scores = [analyzer.polarity_scores(s)['compound'] for s in sentences]
    
    else:
        raise ValueError(f"Unknown method {method!r} (expected 'transformer' or 'vader')")

    return bin_to_arc(scores, arc_length)


def mean_pairwise_l2(arcs):
    """Mean pairwise Euclidean distance between a list of arc vectors. 0.0 if fewer than 2 arcs."""

    # Return 0.0 if fewer than 2 arcs (pairwise diversity is undefined)
    if len(arcs) < 2:
        return 0.0

    # Compute all pairwise distances and return the mean
    arcs_arr = np.array(arcs)
    dists = [np.linalg.norm(arcs_arr[i] - arcs_arr[j]) for i, j in combinations(range(len(arcs_arr)), 2)]

    return float(np.mean(dists))


def position_wise_diversity(arcs):
    """Mean pairwise squared difference at each narrative position, shape (arc_length,)

    sum_t D(t) == mean_pairwise_l2(arcs) ** 2 (per-position decomposition of D_narr)
    """
    arcs_arr = np.array(arcs)

    # Filter out arcs that are None (e.g. too short to produce a full arc)
    if len(arcs_arr) < 2:
        return None

    # Compute all pairwise squared differences at each position and return the mean
    diffs_sq = np.array([(arcs_arr[i] - arcs_arr[j]) ** 2 for i, j in combinations(range(len(arcs_arr)), 2)])
    return diffs_sq.mean(axis=0)


def narrative_diversity(texts, method: str = 'transformer', arc_length: int = ARC_LENGTH,
                         min_sentences: int = MIN_SENTENCES, min_valid_texts: int = 2):
    """D_narr for one group of texts (e.g. multiple generations for one prompt).

    Texts shorter than min_sentences are dropped
    Raises ValueError if fewer than min_valid_texts remain
    """

    # Compute sentiment arcs for each text, filtering out short texts
    arcs = [get_sentiment_arc(t, method=method, arc_length=arc_length, min_sentences=min_sentences) for t in texts]
    arcs = [a for a in arcs if a is not None]

    # Check if there are enough valid texts to compute D_narr
    if len(arcs) < min_valid_texts:
        raise ValueError(f"Only {len(arcs)} valid text(s) (need >= {min_valid_texts}) after filtering short texts")

    # Compute mean pairwise Euclidean distance between arcs
    return mean_pairwise_l2(arcs)
