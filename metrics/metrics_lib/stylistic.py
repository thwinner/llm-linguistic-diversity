"""Stylistic diversity (D_style): 11-feature stylometric distance

D_style combines two components per group of texts (i.e. multiple generations for one prompt):
  - mean pairwise Euclidean distance over 7 z-standardized scalar features
    (sentence length mean/variance, adjective+adverb ratio, mean syllables,
    quotation-mark density, punctuation expressivity, sentence-initial pronoun
    ratio). The z-standardization uses a single global mean/SD per feature,
    computed once over every valid text in the whole comparison set, so a
    group's scalar distances sit on a fixed shared scale rather than a
    per-group one.
  - mean pairwise cosine distance over the function-word frequency
    distribution, taken only over texts that contain at least one function
    word (texts with none are excluded).

Both raw components are winsorized (IQR) and min-max normalized across all
groups being compared, then averaged 0.5/0.5 into D_style. This metric needs
the whole batch of groups at once.

A component is np.nan when it is undefined for a group: the scalar
component when the group has fewer than 2 valid texts, the function-word
component when fewer than 2 of its texts carry a function-word distribution.
D_style is np.nan whenever either component is.

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

import re
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine
from ._common import get_spacy_nlp

# Minimum number of sentences required to compute stylistic features for a text.
MIN_SENTENCES = 3

# A "word" for the length features = a maximal run of Unicode letters (no
# digits, no underscore, no punctuation). Sentence and paragraph length are
# counted in these alphabetic words
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _alpha_words(text):
    """List of alphabetic word tokens in ``text`` (letters only)."""
    return _WORD_RE.findall(text)


# Quotation marks counted for the quotation-mark-density feature: straight and
# curly doubles, German low-9, and guillemets, opening and closing
# -> so every quote character a dialogue turn carries is counted not just the opener.
QUOTATION_MARKS = ('"', '“', '”', '„', '«', '»')


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

    # NLTK is only used as the function-word (stopword) list
    # sentence/word tokenization below uses spaCy
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

    # Iterate through each character in the word to count syllables
    for ch in word:
        is_vowel = ch in vowels

        # Increment syllable count when a vowel group starts (current char is a vowel and previous char was not)
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    # Handle silent 'e' at the end of words (e.g., "make" has 1 syllable, not 2)
    if word.endswith('e') and count > 1:
        count -= 1

    return max(count, 1)


def extract_style_features(text: str, min_sentences: int = MIN_SENTENCES, nlp=None, function_words=None):
    """11-feature stylistic vector for one text or None if too short
    (< min_sentences). The returned dict's 'fw_dist' entry is None when the text
    contains none of the function words (the scalar features are still valid)."""

    # Ensure NLTK stopwords are downloaded (used as the function-word list)
    _ensure_nltk_resources()
    import textstat

    # Ensure spaCy NLP pipeline and function words are available
    nlp = nlp if nlp is not None else get_spacy_nlp()
    if nlp is None:
        raise RuntimeError("extract_style_features requires spaCy (en_core_web_sm) to be installed")

    # Ensure function words are available (NLTK stopwords)
    function_words = function_words if function_words is not None else get_function_words()

    # Tokenize sentences and words via spaCy.
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

    # Feature 1: function-word frequency distribution. A text with no function
    # word at all gets fw_dist = None and is dropped from the FW component downstream
    fw_counts = np.array([words_alpha.count(fw) for fw in function_words], dtype=float)
    fw_total = fw_counts.sum()
    fw_dist = fw_counts / fw_total if fw_total > 0 else None

    # Feature 2 & 3: sentence length mean & variance (alphabetic words per
    # sentence, not tokenizer tokens)
    sent_lengths = [len(_alpha_words(s.text)) for s in sentences]
    sent_len_mean = np.mean(sent_lengths)
    sent_len_var = np.var(sent_lengths)

    # Feature 4: Flesch Reading Ease
    flesch = textstat.flesch_reading_ease(text)

    # Feature 5: comma density (commas per sentence)
    comma_density = text.count(',') / len(sentences)

    # Feature 6: (adjective + adverb) to content-word ratio via spaCy
    # Content words are defined as nouns, verbs, adjectives, adverbs and proper nouns.
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
    adj_adv_pos = {'ADJ', 'ADV'}

    # Count content words and adjectives/adverbs in the text using spaCy's part-of-speech tagging
    content_tokens = [t for t in doc if t.pos_ in content_pos]
    adj_adv_tokens = [t for t in doc if t.pos_ in adj_adv_pos]
    adj_adv_ratio = len(adj_adv_tokens) / (len(content_tokens) + 1e-9)

    # Feature 7: mean word length in syllables
    mean_syllables = np.mean([count_syllables(w) for w in words_alpha])

    # Feature 8: quotation-mark density
    quote_marks = sum(text.count(q) for q in QUOTATION_MARKS)
    dialogue_ratio = quote_marks / (len(sentences) + 1e-9)

    # Feature 9: punctuation expressivity (?, !, ellipses per sentence)
    expressive_punct = text.count('?') + text.count('!') + text.count('…') + text.count('...')
    punct_expressivity = expressive_punct / (len(sentences) + 1e-9)

    # Feature 10: mean paragraph length (alphabetic words per paragraph, not
    # tokenizer tokens)
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 0] or [text]
    mean_para_length = np.mean([len(_alpha_words(p)) for p in paragraphs])

    # Feature 11: sentence-initial pronoun ratio -> checked on the first
    # *alphabetic* token of each sentence (skips leading quotes, dashes, other
    # punctuation)
    pron_starts = 0
    for s in sentences:
        first_alpha = next((tok for tok in s if tok.is_alpha), None)
        if first_alpha is not None and first_alpha.pos_ == 'PRON':
            pron_starts += 1
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


def compute_d_style_components(features_list, scalar_features=DEFAULT_SCALAR_FEATURES, *, mu, sigma):
    """Raw (unnormalized) D_style components for one group of already-extracted features.

    features_list: list of dicts from extract_style_features() (None entries
      already filtered out).
    mu, sigma: global per-feature mean / std for z-standardizing the scalar
      features (both required), computed once over every valid text in the
      whole comparison set by stylistic_diversity() and passed in, so a group's
      scalar distances sit on a fixed shared scale.

    Returns (scalar_component, fw_component). Either is np.nan when undefined:
      - scalar_component: fewer than 2 texts in the group
      - fw_component: fewer than 2 texts with a function-word distribution
        (texts without any function word are excluded)
    These must be normalized across all groups before combining into D_style
    (see stylistic_diversity()).
    """

    n = len(features_list)

    # 1. Scalar component: mean pairwise Euclidean distance over z-scores

    # If there are fewer than 2 texts, the scalar component is undefined (NaN)
    if n < 2:
        scalar_component = float("nan")
    else:
        scalar_matrix = np.array(
            [[f[name] for name in scalar_features] for f in features_list], dtype=float
        )

        # Z-standardize the scalar features using the provided global mean and std
        mu = np.asarray(mu, dtype=float)
        sigma = np.array(sigma, dtype=float, copy=True)
        sigma[sigma == 0] = 1.0  # avoid division by zero for constant features

        z = (scalar_matrix - mu) / sigma
        pairs = list(combinations(range(n), 2))
        scalar_component = float(np.mean([np.linalg.norm(z[i] - z[j]) for i, j in pairs]))

    # 2. Function-word component: mean pairwise cosine distance
    # Only texts that actually contain a function word (fw_dist is not None)
    fw_vectors = [f['fw_dist'] for f in features_list if f.get('fw_dist') is not None]

    # If there are fewer than 2 texts with a function-word distribution, the function-word component is undefined (NaN)
    if len(fw_vectors) < 2:
        fw_component = float("nan")
    else:
        fw_matrix = np.stack(fw_vectors)
        fw_pairs = list(combinations(range(len(fw_vectors)), 2))
        fw_component = float(np.mean([cosine(fw_matrix[i], fw_matrix[j]) for i, j in fw_pairs]))

    return scalar_component, fw_component


def _iqr_bounds(values, k=1.5):
    values = np.asarray(values, dtype=float)

    # Q1 / Q3 over the defined values only (NaN groups are ignored)
    q1, q3 = np.nanpercentile(values, [25, 75])

    # Lower / upper winsorization fences
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def _winsorized_minmax(values, k=1.5):
    """Winsorize at IQR fences, then min-max to [0, 1]. NaN entries (groups with
    an undefined component) stay NaN and are ignored by the fences and span."""
    values = np.asarray(values, dtype=float)

    # Nothing defined -> nothing to normalize
    if not np.any(np.isfinite(values)):
        return values

    lower, upper = _iqr_bounds(values, k)

    # Clip to the fences, NaN stays NaN
    clipped = np.clip(values, lower, upper)

    # Min-max normalize the clipped values to [0, 1], ignoring NaN
    lo, hi = np.nanmin(clipped), np.nanmax(clipped)
    span = hi - lo

    # If the span is zero (all defined values equal), map them to 0, keep NaN
    if span == 0:
        out = np.zeros_like(clipped)
        out[np.isnan(clipped)] = np.nan
        return out

    # Min-max normalize the clipped values to [0, 1]
    return (clipped - lo) / span


def stylistic_diversity(texts_by_group, scalar_features=DEFAULT_SCALAR_FEATURES,
                         min_sentences=MIN_SENTENCES, winsorize_k=1.5, nlp=None,
                         return_components=False):
    """D_style per group, normalized across the whole batch of groups

    texts_by_group: dict[group_id, list[str]] (e.g. {prompt_id: [generations]})
    Returns dict[group_id, float]

    If return_components=True, instead returns (d_style, scalar_norm, fw_norm):
    three dict[group_id, float]

    Texts shorter than min_sentences are dropped before computing a group's raw
    components. The scalar features are z-standardized with a single global
    mean/SD per feature, taken over every valid text in the whole batch. A group
    with fewer than 2 valid texts (or fewer than 2 texts carrying a function-word
    distribution) gets a NaN component, which propagates to a NaN D_style.
    """

    # Ensure spaCy NLP pipeline and function words are available
    nlp = nlp if nlp is not None else get_spacy_nlp()
    function_words = get_function_words()

    # Extract features once per text, kept grouped
    features_by_group, all_features = {}, []
    for group_id, texts in texts_by_group.items():
        features = [extract_style_features(t, min_sentences, nlp, function_words) for t in texts]
        features = [f for f in features if f is not None]
        features_by_group[group_id] = features
        all_features.extend(features)

    group_ids = list(features_by_group)

    # No valid text anywhere -> every group's D_style is undefined
    if not all_features:
        nan_dict = {gid: float("nan") for gid in group_ids}

        if not return_components:
            return nan_dict
        return nan_dict, dict(nan_dict), dict(nan_dict)

    # Global z-standardization params: one mean / SD per scalar feature over
    # every valid text in the whole comparison set (not per group)
    global_matrix = np.array(
        [[f[name] for name in scalar_features] for f in all_features], dtype=float
    )
    global_mu = global_matrix.mean(axis=0)
    global_sigma = global_matrix.std(axis=0)

    # Raw components per group, using the global standardization
    scalar_raw, fw_raw = [], []
    for group_id in group_ids:
        scalar_c, fw_c = compute_d_style_components(
            features_by_group[group_id], scalar_features, mu=global_mu, sigma=global_sigma
        )
        scalar_raw.append(scalar_c)
        fw_raw.append(fw_c)

    # Normalize the raw components across all groups using winsorized min-max normalization
    scalar_norm = _winsorized_minmax(scalar_raw, winsorize_k)
    fw_norm = _winsorized_minmax(fw_raw, winsorize_k)

    # Combine the normalized components into the final D_style score (0.5/0.5 weighting)
    d_style = 0.5 * scalar_norm + 0.5 * fw_norm

    # Return the D_style scores as a dict[group_id, float]
    d_style_dict = {gid: float(d) for gid, d in zip(group_ids, d_style)}
    if not return_components:
        return d_style_dict

    # Return the normalized components for further analysis if requested
    scalar_norm_dict = {gid: float(s) for gid, s in zip(group_ids, scalar_norm)}
    fw_norm_dict = {gid: float(f) for gid, f in zip(group_ids, fw_norm)}
    
    return d_style_dict, scalar_norm_dict, fw_norm_dict
