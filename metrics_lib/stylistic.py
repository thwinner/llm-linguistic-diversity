"""Stylistic diversity (D_style): 11-feature stylometric distance.

D_style combines two components per group of texts (i.e. multiple generations
for one prompt):
  - mean pairwise Euclidean distance over 7 z-standardized scalar features
    (sentence length mean/variance, adjective+adverb ratio, mean syllables,
    dialogue ratio, punctuation expressivity, sentence-initial pronoun ratio)
  - mean pairwise cosine distance over the function-word frequency distribution

Both raw components are winsorized (IQR) and min-max normalized across all
groups being compared, then averaged 0.5/0.5 into D_style. This metric needs 
the whole batch of groups at once. 
Call stylistic_diversity() once with every group you want scored, not group by group.

    from metrics_lib.stylistic import stylistic_diversity
    d_style = stylistic_diversity({"prompt_0": texts_0, "prompt_1": texts_1, ...})

    # for a weighting-sensitivity analysis (recombining with weights other than 0.5/0.5):
    d_style, scalar_norm, fw_norm = stylistic_diversity(texts_by_group, return_components=True)
    d_style_w03 = {gid: 0.3 * scalar_norm[gid] + 0.7 * fw_norm[gid] for gid in scalar_norm}

The default scalar feature set (DEFAULT_SCALAR_FEATURES) drops 3 features
(flesch_reading_ease, comma_density, mean_para_length) found redundant
(|Spearman r| > 0.6 with another feature) on the original storytelling corpus. 
"""

from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine

from ._common import get_spacy_nlp

# Minimum number of sentences required to compute stylistic features for a text.
MIN_SENTENCES = 3


# Contraction fragments to exclude from NLTK's stopword list when computing
_CONTRACTION_FRAGMENTS = {
    'ain', 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven',
    'isn', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', 'd', 'll', 'm', 'ma', 'o', 're', 's', 't', 've', 'y',
}


# All stylistic features extracted by extract_style_features() -> 11 total
ALL_SCALAR_FEATURES = [
    'sent_len_mean', 'sent_len_var', 'flesch_reading_ease',
    'comma_density', 'adj_adv_ratio', 'mean_syllables',
    'dialogue_ratio', 'punct_expressivity', 'mean_para_length', 'sent_start_pron_ratio',
]

# Redundant features removed after correlation analysis on storytelling task
REDUNDANT_FEATURES = ['flesch_reading_ease', 'comma_density', 'mean_para_length']

# Default scalar features used for D_style (7 features)
DEFAULT_SCALAR_FEATURES = [f for f in ALL_SCALAR_FEATURES if f not in REDUNDANT_FEATURES]


# Cached spaCy NLP pipeline to avoid repeated model loading
@lru_cache(maxsize=1)
def _ensure_nltk_resources():
    import nltk

    # NLTK is only used as the function-word (stopword) list; sentence/word
    # tokenization below uses spaCy throughout, for consistency with the
    # spaCy-based POS features (adj_adv_ratio, sent_start_pron_ratio).
    nltk.download('stopwords', quiet=True)
    return True


# Cached spaCy NLP pipeline to avoid repeated model loading
@lru_cache(maxsize=1)
def get_function_words():
    """NLTK's English stopword list (Bird, Klein & Loper, 2009), restricted to
    standalone alphabetic entries (drops contraction fragments like "don", "ve")."""

    # Ensure NLTK resources are downloaded
    _ensure_nltk_resources()
    from nltk.corpus import stopwords

    # Return a sorted tuple of function words (stopwords) for consistent ordering
    return tuple(sorted(
        w for w in stopwords.words('english')
        if w.isalpha() and w not in _CONTRACTION_FRAGMENTS
    ))


def count_syllables(word: str) -> int:
    """Heuristic syllable counter (vowel-group based)"""

    # Count contiguous vowel groups as syllables
    word = word.lower()
    vowels = 'aeiouy'
    count = 0
    prev_was_vowel = False

    for ch in word:
        is_vowel = ch in vowels

        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    # Handle silent 'e' at the end of words (e.g., "make" has 1 syllable, not 2)
    if word.endswith('e') and count > 1:
        count -= 1

    return max(count, 1)


def extract_style_features(text: str, min_sentences: int = MIN_SENTENCES, nlp=None, function_words=None):
    """11-feature stylistic vector for one text or None if too short (< min_sentences)"""

    # Ensure NLTK stopwords are downloaded (used as the function-word list)
    _ensure_nltk_resources()
    import textstat

    # Ensure spaCy NLP pipeline and function words are available
    nlp = nlp if nlp is not None else get_spacy_nlp()
    if nlp is None:
        raise RuntimeError("extract_style_features requires spaCy (en_core_web_sm) to be installed")
    function_words = function_words if function_words is not None else get_function_words()

    # Tokenize sentences and words via spaCy, so every sentence- and word-level
    # feature below shares the same tokenization (previously sentence length,
    # dialogue/punctuation ratios etc. used NLTK's Punkt tokenizer while the
    # pronoun-start feature used spaCy's, i.e. two different sentence counts).
    doc = nlp(text)
    sentences = [s for s in doc.sents if len(s.text.strip()) > 0]

    # Check if the text has enough sentences to compute stylistic features
    if len(sentences) < min_sentences:
        return None

    # Tokenize words and filter to alphabetic words for stylistic analysis
    words = [t.text for t in doc]
    words_alpha = [w.lower() for w in words if w.isalpha()]

    # If there are no alphabetic words, return None to avoid division by zero
    if len(words_alpha) == 0:
        return None

    # Feature 1: function-word frequency distribution
    fw_counts = np.array([words_alpha.count(fw) for fw in function_words], dtype=float)
    fw_dist = fw_counts / (fw_counts.sum() + 1e-9)

    # Feature 2 & 3: sentence length mean & variance (in words)
    sent_lengths = [len(s) for s in sentences]
    sent_len_mean = np.mean(sent_lengths)
    sent_len_var = np.var(sent_lengths)

    # Feature 4: Flesch Reading Ease
    flesch = textstat.flesch_reading_ease(text)

    # Feature 5: comma density (commas per sentence)
    comma_density = text.count(',') / len(sentences)

    # Feature 6: (adjective + adverb) to content-word ratio via spaCy
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
    adj_adv_pos = {'ADJ', 'ADV'}

    content_tokens = [t for t in doc if t.pos_ in content_pos]
    adj_adv_tokens = [t for t in doc if t.pos_ in adj_adv_pos]
    adj_adv_ratio = len(adj_adv_tokens) / (len(content_tokens) + 1e-9)

    # Feature 7: mean word length in syllables
    mean_syllables = np.mean([count_syllables(w) for w in words_alpha])

    # Feature 8: dialogue ratio (opening quotation marks per sentence, proxy for dialogue turns)
    dialogue_turns = text.count('"') + text.count('“')
    dialogue_ratio = dialogue_turns / (len(sentences) + 1e-9)

    # Feature 9: punctuation expressivity (?, !, ellipses per sentence)
    expressive_punct = text.count('?') + text.count('!') + text.count('…') + text.count('...')
    punct_expressivity = expressive_punct / (len(sentences) + 1e-9)

    # Feature 10: mean paragraph length (words per paragraph)
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 0] or [text]
    mean_para_length = np.mean([len(nlp.tokenizer(p)) for p in paragraphs])

    # Feature 11: sentence-initial pronoun ratio
    pron_starts = sum(1 for s in sentences if len(s) > 0 and s[0].pos_ == 'PRON')
    sent_start_pron_ratio = pron_starts / (len(sentences) + 1e-9)

    return {
        'fw_dist': fw_dist,
        'sent_len_mean': sent_len_mean,
        'sent_len_var': sent_len_var,
        'flesch_reading_ease': flesch,
        'comma_density': comma_density,
        'adj_adv_ratio': adj_adv_ratio,
        'mean_syllables': mean_syllables,
        'dialogue_ratio': dialogue_ratio,
        'punct_expressivity': punct_expressivity,
        'mean_para_length': mean_para_length,
        'sent_start_pron_ratio': sent_start_pron_ratio,
    }


def compute_d_style_components(features_list, scalar_features=DEFAULT_SCALAR_FEATURES):
    """Raw (unnormalized) D_style components for one group of already-extracted features.

    features_list: list of dicts from extract_style_features() (None entries
    already filtered out) 
    Returns (scalar_component, fw_component)
    These must be normalized across all groups before combining into D_style, see
    stylistic_diversity()
    """

    n = len(features_list)
    if n < 2:
        return 0.0, 0.0

    scalar_matrix = np.array([[f[name] for name in scalar_features] for f in features_list], dtype=float)

    # Mean and standard deviation for z-standardization of scalar features
    mu = scalar_matrix.mean(axis=0)
    sigma = scalar_matrix.std(axis=0)

    # Avoid division by zero in case of zero standard deviation
    sigma[sigma == 0] = 1.0

    # Z-standardize scalar features for pairwise distance computation
    z = (scalar_matrix - mu) / sigma

    # Compute all pairwise Euclidean distances for z-standardized scalar features
    pairs = list(combinations(range(n), 2))
    scalar_dists = [np.linalg.norm(z[i] - z[j]) for i, j in pairs]

    # Compute all pairwise cosine distances for function-word frequency distributions
    fw_matrix = np.stack([f['fw_dist'] for f in features_list])
    fw_dists = [cosine(fw_matrix[i], fw_matrix[j]) for i, j in pairs]
    fw_dists = [d if not np.isnan(d) else 0.0 for d in fw_dists]

    return float(np.mean(scalar_dists)), float(np.mean(fw_dists))


def _iqr_bounds(values, k=1.5):
    values = np.asarray(values, dtype=float)

    # Compute the first (Q1) and third (Q3) quartiles and the interquartile range (IQR)
    q1, q3 = np.percentile(values, [25, 75])

    # Compute the lower and upper bounds for winsorization based on the IQR and the specified k value
    iqr = q3 - q1

    # Return the lower and upper bounds for winsorization
    return q1 - k * iqr, q3 + k * iqr


def _winsorized_minmax(values, k=1.5):
    values = np.asarray(values, dtype=float)

    # Compute the lower and upper bounds for winsorization using the IQR method
    lower, upper = _iqr_bounds(values, k)

    # Winsorize the values by clipping them to the computed lower and upper bounds
    clipped = np.clip(values, lower, upper)

    # Compute the span (range) of the clipped values
    span = clipped.max() - clipped.min()

    # If the span is zero (all values are the same), return an array of zeros to avoid division by zero
    if span == 0:
        return np.zeros_like(clipped)

    # Min-max normalize the clipped values to the range [0, 1]
    return (clipped - clipped.min()) / span


def stylistic_diversity(texts_by_group, scalar_features=DEFAULT_SCALAR_FEATURES,
                         min_sentences=MIN_SENTENCES, winsorize_k=1.5, nlp=None,
                         return_components=False):
    """D_style per group, normalized across the whole batch of groups

    texts_by_group: dict[group_id, list[str]] (e.g. {prompt_id: [generations]})
    Returns dict[group_id, float].
    If return_components=True, instead returns (d_style, scalar_norm, fw_norm) -- three
    dict[group_id, float], for e.g. a weighting-sensitivity analysis that recombines
    scalar_norm/fw_norm with weights other than the default 0.5/0.5.
    Texts shorter than min_sentences are dropped before computing a group's raw components.
    """

    # Ensure spaCy NLP pipeline and function words are available
    nlp = nlp if nlp is not None else get_spacy_nlp()
    function_words = get_function_words()

    # Compute raw stylistic components for each group of texts
    scalar_raw, fw_raw, group_ids = [], [], []
    for group_id, texts in texts_by_group.items():
        features = [extract_style_features(t, min_sentences, nlp, function_words) for t in texts]
        features = [f for f in features if f is not None]
        scalar_c, fw_c = compute_d_style_components(features, scalar_features)
        scalar_raw.append(scalar_c)
        fw_raw.append(fw_c)
        group_ids.append(group_id)

    # Normalize the raw components across all groups using winsorized min-max normalization
    scalar_norm = _winsorized_minmax(scalar_raw, winsorize_k)
    fw_norm = _winsorized_minmax(fw_raw, winsorize_k)

    # Combine the normalized components into the final D_style score (0.5/0.5 weighting)
    d_style = 0.5 * scalar_norm + 0.5 * fw_norm

    d_style_dict = {gid: float(d) for gid, d in zip(group_ids, d_style)}
    if not return_components:
        return d_style_dict

    scalar_norm_dict = {gid: float(s) for gid, s in zip(group_ids, scalar_norm)}
    fw_norm_dict = {gid: float(f) for gid, f in zip(group_ids, fw_norm)}
    return d_style_dict, scalar_norm_dict, fw_norm_dict
