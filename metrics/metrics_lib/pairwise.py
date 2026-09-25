"""
Pairwise distance matrices for the group-level diversity metrics, plus block
slicing for cross-source comparisons (within-source diversity vs. distance to a
human reference).

The ``*_diversity()`` functions in :mod:`metrics_lib` collapse a group of texts
to one score (the mean over every pairwise distance). This module exposes the
step before that collapse: the full ``n x n`` distance matrix, built from the
same per-text representations and the same distance as the corresponding metric. 
From a matrix you can average whichever subset of pairs you want

    from metrics.metrics_lib.pairwise import distance_matrix, within_between, slices_from_lengths

    texts   = human_texts + gemma_texts + qwen_texts
    slices  = slices_from_lengths([("human", len(human_texts)),
                                   ("gemma", len(gemma_texts)),
                                   ("qwen",  len(qwen_texts))])
    D  = distance_matrix("D_sem", texts)
    wb = within_between(D, slices)

    wb.within["gemma"]              # gemma's within-group distances (== the D_sem group view)
    wb.within["human"]             # every within-human distance
    wb.between[("human", "gemma")]  # every human-vs-gemma distance (distance to human reference)

Reference / control split (the leave-one-out scheme): build one matrix per
``(prompt, metric)`` over ``[humans | model_1 | model_2 | ...]`` with the human
texts first. Then

  * ``wb.within["human"]`` is exactly the pooled leave-one-out control distribution
  * ``wb.between[("human", <model>)]`` is that model's distance-to-human
    distribution, scored against the same reference.

Collect those arrays over all prompts to get one credal set per task and
dimension, with "Human (control)" as one of the sources.

If instead you need control and models scored against a reference of the
same fixed size within a replicate, split the human block into a
fixed-size ``H_ref`` / ``H_ctrl`` and repeat over every such split:

    for split_id, bands in reference_control_splits(D, n_human, other_slices, ref_size=3):
        bands["human-human"]   # this split's H_ctrl x H_ref distances
        bands["gemma"]          # this split's gemma x H_ref distances

Undefined pairs (empty token sets, missing entity profile, a text too short for
a sentiment arc) are ``np.nan`` in the matrix and are dropped by
:func:`within_between`.

**D_style** is handled separately -> see :func:`style_distance_matrices`.
Because its two raw components are winsorized + min-max normalized across the
whole batch of groups rather than per pair. Like the credal-set cell in
``metrics/01_discourse_diversity/disc_diversity.ipynb``, the D_style helper here
normalizes pooled *per-pair* values, so it does not reproduce the group-level
``stylistic_diversity()`` scores exactly.

If you need a *per-text* (not per-band) distance to the reference, use
:func:`reference_control_splits_rowwise` instead, which yields every row's
distance-to-reference rather than pooling rows into bands first.
"""

from dataclasses import dataclass
from itertools import combinations
import warnings

import numpy as np
from scipy.spatial.distance import cosine

from ._common import get_spacy_nlp
from .semantic_lexical_syntactic import (
    word_unigrams, pos_bigrams, jaccard_distance,
    _get_sbert_model, _is_nomic, NOMIC_TASK_PREFIX, DEFAULT_SBERT_MODEL,
)
from .discourse import compute_transition_profile, jsd
from .narrative import get_sentiment_arc, ARC_LENGTH, MIN_SENTENCES as NARR_MIN_SENTENCES
from .stylistic import (
    extract_style_features, get_function_words,
    DEFAULT_SCALAR_FEATURES, MIN_SENTENCES as STYLE_MIN_SENTENCES,
)

__all__ = [
    "distance_matrix",
    "semantic_distance_matrix",
    "lexical_distance_matrix",
    "syntactic_distance_matrix",
    "discourse_distance_matrix",
    "narrative_distance_matrix",
    "within_between",
    "WithinBetween",
    "slices_from_lengths",
    "reference_control_splits",
    "reference_control_splits_rowwise",
    "style_distance_matrices",
    "winsor_minmax_fit",
    "winsor_minmax_apply",
]

MATRIX_METRICS = ("D_sem", "D_lex", "D_syn", "D_disc", "D_narr")


# --------------------------------------------------------------------------- #
# Per-metric distance matrices                                               #
# --------------------------------------------------------------------------- #

def semantic_distance_matrix(texts, model_name=DEFAULT_SBERT_MODEL, model=None):
    """D_sem: (n, n) matrix of pairwise cosine distances of text embeddings.

    Same encoder, task prefix and ``normalize_embeddings=True`` as
    :func:`metrics_lib.semantic_diversity`; ``np.nanmean`` of the upper triangle
    reproduces its group score. The diagonal is ``np.nan``.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Use the provided model if given, otherwise load the SBERT model by name.
    model = model if model is not None else _get_sbert_model(model_name)

    # The model's encode() method expects a list of strings, not a generator.
    enc_texts = list(texts)

    # Nomic models expect a task prefix on every input string, so add it here if
    # the model name indicates a Nomic model.
    if _is_nomic(model_name):
        enc_texts = [NOMIC_TASK_PREFIX + t for t in enc_texts]

    # Small batch: WritingPrompts stories run to ~1.5k tokens and the default
    # batch of 32 spikes attention activations enough to swap a 16 GB machine.
    embs = model.encode(enc_texts, show_progress_bar=False, normalize_embeddings=True,
                        batch_size=8)

    # The cosine_similarity() function returns a similarity matrix, so we convert it to a distance matrix.
    dist = 1.0 - cosine_similarity(embs)

    # The diagonal is set to np.nan to indicate that the distance of a text to itself is undefined.
    np.fill_diagonal(dist, np.nan)
    return dist


def _jaccard_matrix(token_sets):
    """(n, n) Jaccard-distance matrix; a pair with two empty sets stays ``np.nan``
    (matching how D_lex / D_syn skip those pairs). Diagonal is ``np.nan``."""
    n = len(token_sets)

    # The distance matrix is initialized with np.nan values
    dist = np.full((n, n), np.nan)

    # Compute the Jaccard distance for each pair of token sets
    for i, j in combinations(range(n), 2):
        if token_sets[i] or token_sets[j]:
            d = jaccard_distance(token_sets[i], token_sets[j])
            dist[i, j] = dist[j, i] = d
    
    return dist


def lexical_distance_matrix(texts):
    """D_lex: (n, n) pairwise Jaccard distance of word-unigram sets."""

    # The word_unigrams function extracts the set of unique words from each text
    return _jaccard_matrix([word_unigrams(t) for t in texts])


def syntactic_distance_matrix(texts, nlp=None):
    """D_syn: (n, n) pairwise Jaccard distance of POS-tag-bigram sets."""

    # The nlp argument allows the caller to provide a pre-loaded spaCy model, which can save time if multiple calls are made
    nlp = nlp if nlp is not None else get_spacy_nlp()

    # The pos_bigrams function extracts the POS-tag bigrams from each text using the provided spaCy model
    return _jaccard_matrix([pos_bigrams(t, nlp=nlp) for t in texts])


def discourse_distance_matrix(texts, use_trigrams=True, alpha_trigram=0.3, nlp=None, weight_scheme=None):
    """D_disc: (n, n) pairwise entity-grid JSD.

    Per pair: ``JSD_bigram``, blended with ``JSD_trigram`` as
    ``(1 - alpha_trigram) * JSD_bigram + alpha_trigram * JSD_trigram`` when both
    texts carry a trigram profile. Texts with no usable entity profile give an
    all-``np.nan`` row/column.
    """

    # The compute_transition_profile function computes the entity-grid transition profile for each text, returning both bigram and trigram profiles
    profiles = [compute_transition_profile(t, use_trigrams=use_trigrams, nlp=nlp, weight_scheme=weight_scheme)
                for t in texts]
    bi = [p[0] for p in profiles]
    tri = [p[1] for p in profiles]

    # The distance matrix is initialized with np.nan values
    n = len(texts)
    dist = np.full((n, n), np.nan)

    # Compute the JSD distance for each pair of texts
    for i, j in combinations(range(n), 2):
        if bi[i] is None or bi[j] is None:
            continue
        d = jsd(bi[i], bi[j])
        if use_trigrams and tri[i] is not None and tri[j] is not None:
            d = (1 - alpha_trigram) * d + alpha_trigram * jsd(tri[i], tri[j])

        dist[i, j] = dist[j, i] = float(d)
    return dist


def narrative_distance_matrix(texts, method="transformer", arc_length=ARC_LENGTH, min_sentences=NARR_MIN_SENTENCES, arcs=None):
    """D_narr: (n, n) pairwise Euclidean distance of sentiment-arc vectors.

    Texts shorter than ``min_sentences`` have no arc and give an all-``np.nan``
    row/column. Pass ``arcs`` (a list aligned with ``texts``, ``None`` for a
    text without an arc) to reuse cached sentiment arcs and skip re-scoring.
    """

    # If arcs are not provided, compute them for each text using the specified method, arc length and minimum sentence count. 
    if arcs is None:
        arcs = [get_sentiment_arc(t, method=method, arc_length=arc_length, min_sentences=min_sentences)
                for t in texts]

    # The distance matrix is initialized with np.nan values
    n = len(arcs)
    dist = np.full((n, n), np.nan)

    # Compute the Euclidean distance for each pair of arcs
    for i, j in combinations(range(n), 2):
        if arcs[i] is None or arcs[j] is None:
            continue
        dist[i, j] = dist[j, i] = float(np.linalg.norm(np.asarray(arcs[i]) - np.asarray(arcs[j])))
    return dist


_MATRIX_FUNCS = {
    "D_sem": semantic_distance_matrix,
    "D_lex": lexical_distance_matrix,
    "D_syn": syntactic_distance_matrix,
    "D_disc": discourse_distance_matrix,
    "D_narr": narrative_distance_matrix,
}


def distance_matrix(metric, texts, **kwargs):
    """Dispatch to the ``*_distance_matrix`` for ``metric`` (one of
    :data:`MATRIX_METRICS`). ``**kwargs`` pass through. D_style has no single-call
    matrix -> use :func:`style_distance_matrices`."""
    # The function retrieves the appropriate distance matrix function based on the provided metric
    try:
        func = _MATRIX_FUNCS[metric]

    # If the metric is not found in the _MATRIX_FUNCS dictionary, raise a ValueError 
    except KeyError:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {list(MATRIX_METRICS)} "
            f"(for D_style use style_distance_matrices)"
        ) from None
    
    return func(texts, **kwargs)


# --------------------------------------------------------------------------- #
# Block slicing                                                              #
# --------------------------------------------------------------------------- #

# The WithinBetween dataclass is used to store the results of the within_between function. 
# It contains two dictionaries: 'within' and 'between'.
@dataclass
class WithinBetween:
    """Result of :func:`within_between`.

    ``within[label]`` -> 1D array of finite within-block distances (upper triangle of that source's block).
    ``between[(label_a, label_b)]`` -> 1D array of finite cross-block distances
                                    (every ``label_a`` x ``label_b`` pair);
                                    label order follows the slice order.
    """
    within: dict
    between: dict


def slices_from_lengths(named_lengths):
    """``[(label, length), ...]`` -> ``{label: (start, stop)}`` laid out
    contiguously from 0, i.e. the row layout of
    ``distance_matrix(metric, texts_a + texts_b + ...)``."""
    out, pos = {}, 0
    for label, length in named_lengths:
        out[label] = (pos, pos + length)
        pos += length
    return out


def within_between(dist, slices):
    """Split a distance matrix into within-source and between-source distances.

    ``dist``   : (n, n) symmetric matrix; ``np.nan`` marks undefined pairs and the diagonal.
    ``slices`` : ``{label: (start, stop)}`` (e.g. from :func:`slices_from_lengths`)
                 or an iterable of ``(label, start, stop)``.

    ``np.nan`` entries are dropped, so each returned array holds only the finite
    distances actually available for that block.
    """

    # The function checks if the slices argument has an 'items' attribute (i.e., if it is a dictionary)
    if hasattr(slices, "items"):
        items = [(label, a, b) for label, (a, b) in slices.items()]
    else:
        items = [tuple(x) for x in slices]

    # Initializes two dictionaries: 'within' and 'between'
    within = {}
    for label, a, b in items:
        block = np.asarray(dist)[a:b, a:b]
        vals = block[np.triu_indices(b - a, k=1)]
        within[label] = vals[np.isfinite(vals)]

    between = {}
    for (label_i, ai, bi), (label_j, aj, bj) in combinations(items, 2):
        block = np.asarray(dist)[ai:bi, aj:bj].ravel()
        between[(label_i, label_j)] = block[np.isfinite(block)]

    return WithinBetween(within=within, between=between)


def reference_control_splits(dist, n_human, other_slices, ref_size, n_splits=None, seed=0):
    """Fixed-size human reference/control splits, matched against the same
    ``H_ref`` for every other source.

    Here every band in a given split is scored against the exact same, fixed-size
    ``H_ref``, so "Human (control)" and each model are directly comparable
    within that split

    ``dist``         : (n, n) distance matrix with the human block in rows/cols
                        ``[0, n_human)`` (the layout ``distance_matrix`` /
                        :func:`slices_from_lengths` produce).
    ``n_human``      : size of the human block.
    ``other_slices`` : ``{label: (start, stop)}`` for the non-human segments
                        (e.g. one entry per model).
    ``ref_size``     : size of ``H_ref`` drawn from the human block; the
                        remaining ``n_human - ref_size`` humans are ``H_ctrl``.
    ``n_splits``     : ``None`` (or >= ``C(n_human, ref_size)``) enumerates
                        every combination of ``H_ref`` exactly once
                        (``split_id`` = 0, 1, ...); otherwise draws that many
                        combinations without replacement (seeded).

    Yields ``(split_id, bands)`` where ``bands`` is
    ``{'human-human': H_ctrl x H_ref distances, label: <label> x H_ref distances, ...}``,
    ``np.nan`` dropped.
    """

    # The function converts the input distance matrix to a NumPy array for easier manipulation
    dist = np.asarray(dist)
    all_refs = list(combinations(range(n_human), ref_size))

    # If n_splits is specified and less than the total number of combinations, randomly select n_splits combinations 
    # without replacement using the provided seed for reproducibility.
    if n_splits is not None and n_splits < len(all_refs):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(all_refs), size=n_splits, replace=False)
        all_refs = [all_refs[i] for i in chosen]

    # The function iterates over each reference split, computes the control indices and extracts the relevant 
    # distance blocks for each band. It yields the split ID and the corresponding bands with finite distances.
    for split_id, ref_idx in enumerate(all_refs):
        ref_idx = np.asarray(ref_idx)
        ctrl_idx = np.asarray([i for i in range(n_human) if i not in ref_idx])

        bands = {}
        block = dist[np.ix_(ctrl_idx, ref_idx)].ravel()
        bands["human-human"] = block[np.isfinite(block)]

        for label, (a, b) in other_slices.items():
            block = dist[a:b, :][:, ref_idx].ravel()
            bands[label] = block[np.isfinite(block)]

        yield split_id, bands


def reference_control_splits_rowwise(dist, n_human, ref_size, n_splits=None, seed=0):
    """Like :func:`reference_control_splits`, but keeps every row's own
    distance-to-reference instead of pooling rows into per-band arrays.

    Enumerates the same fixed-size ``H_ref`` splits of the human block
    (``[0, n_human)``). For each split, every row of ``dist`` (human or not)
    gets its mean and nearest (min) finite distance to that split's ``H_ref``
    columns. A human row that is itself part of ``H_ref`` this split has no
    well-defined "distance to reference" (it is the reference): for a human 
    control text, average only over the splits where it lands in ``H_ctrl``; 
    a non-human row is never part of ``H_ref`` and can be averaged over every split.

    ``dist``     : (n, n) distance matrix, human block in rows/cols
                   ``[0, n_human)`` (as :func:`reference_control_splits`).
    ``n_human``  : size of the human block.
    ``ref_size`` : size of ``H_ref`` drawn from the human block.
    ``n_splits`` : as in :func:`reference_control_splits`.

    Yields ``(split_id, ref_idx, row_mean, row_min)``:

    - ``ref_idx``   : this split's ``H_ref`` indices into ``[0, n_human)``.
    - ``row_mean``  : length-``n`` array, each row's mean finite distance to
                      the ``ref_idx`` columns (``np.nan`` if none finite).
    - ``row_min``   : length-``n`` array, each row's nearest (min) finite
                      distance to the ``ref_idx`` columns (``np.nan`` if none
                      finite) -- a robustness variant of ``row_mean`` for a
                      small ``ref_size``, less sensitive to a noisy reference
                      but noisier itself (single nearest neighbor).
    """

    # The function converts the input distance matrix to a NumPy array for easier manipulation
    dist = np.asarray(dist)

    # The function generates all possible combinations of reference indices from the human block 
    all_refs = list(combinations(range(n_human), ref_size))

    # If n_splits is specified and less than the total number of combinations, randomly select n_splits combinations 
    # without replacement using the provided seed for reproducibility.
    if n_splits is not None and n_splits < len(all_refs):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(all_refs), size=n_splits, replace=False)
        all_refs = [all_refs[i] for i in chosen]

    # The function iterates over each reference split, computes the mean and minimum finite distances for each 
    # row to the reference columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-nan row -> nan, expected
        for split_id, ref_idx in enumerate(all_refs):
            ref_idx = np.asarray(ref_idx)
            block = dist[:, ref_idx]
            row_mean = np.nanmean(block, axis=1)
            row_min = np.nanmin(block, axis=1)
            row_min[~np.isfinite(row_min)] = np.nan
            yield split_id, ref_idx, row_mean, row_min


# --------------------------------------------------------------------------- #
# D_style (batch-normalized, per-pair)                                       #
# --------------------------------------------------------------------------- #

def winsor_minmax_fit(values, k=1.5):
    """Fit IQR winsorization + min-max on a pooled 1D sample of raw distances.

    Returns a params dict for :func:`winsor_minmax_apply`. 
    ``k=None`` skips winsorization
    """

    # The function converts the input values to a NumPy array of type float and filters out any non-finite values
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    # If the filtered array is empty, return a dictionary with infinite fences and NaN values for the min and max
    if v.size == 0:
        return {"lo_fence": -np.inf, "hi_fence": np.inf, "lo": np.nan, "hi": np.nan}

    # If k is None, set the lower and upper fences to negative and positive infinity, respectively. 
    if k is None:
        lo_fence, hi_fence = -np.inf, np.inf

    # Otherwise, compute the first and third quartiles (Q1 and Q3) of the filtered array, calculate the interquartile range (IQR) 
    # and set the lower and upper fences based on the IQR and the specified k value.
    else:
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - k * iqr, q3 + k * iqr

    # Clip the filtered array to the lower and upper fences
    clipped = np.clip(v, lo_fence, hi_fence)

    return {"lo_fence": float(lo_fence), "hi_fence": float(hi_fence),
            "lo": float(clipped.min()), "hi": float(clipped.max())}


def winsor_minmax_apply(x, params):
    """Apply :func:`winsor_minmax_fit` params to ``x`` (array or matrix).
    Non-finite entries stay ``np.nan``; a zero span maps every finite entry to 0.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)

    # Clip the input array to the lower and upper fences specified in the params dictionary
    clipped = np.clip(x, params["lo_fence"], params["hi_fence"])
    span = params["hi"] - params["lo"]

    # The function initializes an output array with the same shape as the input array, filled with NaN values.
    out = np.full(x.shape, np.nan)

    # If the span is not finite or is zero, set all finite entries in the output array to 0. 
    # Otherwise, normalize the clipped values to the range [0, 1] based on the specified min and max values in the params dictionary.
    if not np.isfinite(span) or span == 0:
        out[finite] = 0.0
        return out
    
    out[finite] = (clipped[finite] - params["lo"]) / span
    return out


def _global_scalar_mu_sigma(features, scalar_features):
    
    mat = np.array([[f[name] for name in scalar_features] for f in features], dtype=float)

    # The function calculates the mean and standard deviation of the input features for each scalar feature.
    return mat.mean(axis=0), mat.std(axis=0)


def _style_raw_pair_matrices(features_list, *, mu, sigma, scalar_features):
    """(scalar_raw, fw_raw): two (n, n) matrices of raw per-pair style distances.

    ``features_list``: ``extract_style_features`` output per text, ``None`` for a
    too-short text. A ``None`` text -> ``np.nan`` row/col in both; a text without
    any function word -> ``np.nan`` row/col in ``fw_raw`` only.
    """

    # Calculate mu and sigma for the scalar features across the features_list
    n = len(features_list)
    mu = np.asarray(mu, dtype=float)
    sigma = np.array(sigma, dtype=float, copy=True)
    sigma[sigma == 0] = 1.0

    # Normalize the scalar features and extract the function word distributions for each text in the features_list
    z = [None if f is None
         else (np.array([f[name] for name in scalar_features], dtype=float) - mu) / sigma
         for f in features_list]
    fw = [None if (f is None or f.get("fw_dist") is None)
          else np.asarray(f["fw_dist"], dtype=float)
          for f in features_list]

    # Initialize two (n, n) matrices for the raw scalar and function word distances, filled with NaN values
    scalar_raw = np.full((n, n), np.nan)
    fw_raw = np.full((n, n), np.nan)

    # Compute the pairwise distances for the scalar and function word features
    for i, j in combinations(range(n), 2):
        if z[i] is not None and z[j] is not None:
            scalar_raw[i, j] = scalar_raw[j, i] = float(np.linalg.norm(z[i] - z[j]))

        if fw[i] is not None and fw[j] is not None:
            fw_raw[i, j] = fw_raw[j, i] = float(np.nan_to_num(cosine(fw[i], fw[j])))

    return scalar_raw, fw_raw


def style_distance_matrices(texts_by_segment, nlp=None, min_sentences=STYLE_MIN_SENTENCES,
                             scalar_features=DEFAULT_SCALAR_FEATURES, winsorize_k=1.5):
    """Batch D_style: one per-pair distance matrix per segment, normalized across
    the whole batch.

    ``texts_by_segment``: ``{segment_id: [texts]}``

    Returns ``{segment_id: (n, n) ndarray}`` of per-pair D_style distances
    (``0.5 * scalar_norm + 0.5 * fw_norm``), ``np.nan`` where undefined.

    The scalar features are z-standardized with one global mean/SD per feature
    over every valid text in the batch (as in :func:`metrics_lib.stylistic_diversity`).
    The two raw components are then winsorized + min-max normalized over the
    pooled per-pair values across all segments -> so this does not reproduce
    the group-level ``stylistic_diversity()`` scores, it is the per-pair analogue
    used for pairing distributions / credal sets. 
    ``winsorize_k=None`` -> plain pooled min-max.
    """
    nlp = nlp if nlp is not None else get_spacy_nlp()
    fw_words = get_function_words()

    # Extract style features for each text in each segment, skipping texts that are too short
    feats_by_seg = {
        sid: [extract_style_features(t, min_sentences, nlp, fw_words) for t in texts]
        for sid, texts in texts_by_segment.items()
    }
    all_feats = [f for fl in feats_by_seg.values() for f in fl if f is not None]

    # If there are no valid features across all segments, return a dictionary with NaN matrices for each segment
    if not all_feats:
        return {sid: np.full((len(fl), len(fl)), np.nan) for sid, fl in feats_by_seg.items()}

    # Compute the global mean and standard deviation for the scalar features across all valid features
    mu, sigma = _global_scalar_mu_sigma(all_feats, scalar_features)

    # Compute the raw pairwise distance matrices for each segment, normalizing the scalar features using the global mean and standard deviation
    raw_by_seg = {
        sid: _style_raw_pair_matrices(fl, mu=mu, sigma=sigma, scalar_features=scalar_features)
        for sid, fl in feats_by_seg.items()
    }

    def _pool(idx):
        parts = [np.asarray(v[idx])[np.triu_indices(v[idx].shape[0], k=1)]
                 for v in raw_by_seg.values() if v[idx].shape[0] >= 2]
        return np.concatenate(parts) if parts else np.array([])

    scalar_params = winsor_minmax_fit(_pool(0), winsorize_k)
    fw_params = winsor_minmax_fit(_pool(1), winsorize_k)

    # Apply the winsorized and min-max normalized values to the raw pairwise distance matrices
    out = {}
    for sid, (scalar_raw, fw_raw) in raw_by_seg.items():
        scalar_norm = winsor_minmax_apply(scalar_raw, scalar_params)
        fw_norm = winsor_minmax_apply(fw_raw, fw_params)
        out[sid] = 0.5 * scalar_norm + 0.5 * fw_norm
        
    return out
