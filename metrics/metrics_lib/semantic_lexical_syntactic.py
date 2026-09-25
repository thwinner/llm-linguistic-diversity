"""
Semantic, lexical and syntactic diversity (D_sem, D_lex, D_syn):
pairwise text-level diversity across a group of texts (multiple generations for the same prompt)

Applied across whole texts within a group instead of across sentences within one text

- D_sem: mean pairwise cosine distance of Nomic Embed Text v1.5 embeddings (full text, up to 8192 tokens)
- D_lex: mean pairwise Jaccard distance of word unigrams
- D_syn: mean pairwise Jaccard distance of POS-tag bigrams

    from metrics_lib.semantic_lexical_syntactic import semantic_diversity, lexical_diversity, syntactic_diversity
    d_sem = semantic_diversity(list_of_texts_for_one_prompt)
    d_lex = lexical_diversity(list_of_texts_for_one_prompt)
    d_syn = syntactic_diversity(list_of_texts_for_one_prompt)
"""

from functools import lru_cache
from itertools import combinations
import numpy as np
from ._common import get_spacy_nlp
from .baseline import word_tokenize_simple

# Default embedding model for semantic diversity
# Nomic Embed Text v1.5 supports inputs up to 8192 tokens, so whole generations are embedded
DEFAULT_SBERT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# Full native context of Nomic Embed Text v1.5.
NOMIC_MAX_SEQ_LENGTH = 8192

# Symmetric task prefix from the Nomic model card, for similarity between texts of
# the same kind (no query/document asymmetry)
NOMIC_TASK_PREFIX = "clustering: "

# Helper function to check if the model name corresponds to a Nomic model
def _is_nomic(model_name):
    return bool(model_name) and "nomic" in model_name.lower()


# Cached SBERT model loader to avoid reloading the model multiple times
@lru_cache(maxsize=1)
def _get_sbert_model(model_name=DEFAULT_SBERT_MODEL):
    from sentence_transformers import SentenceTransformer

    # Nomic ships custom modelling code and needs trust_remote_code=True.
    model = SentenceTransformer(model_name, trust_remote_code=_is_nomic(model_name))

    # Use the full 8192-token context instead of the packaged default.
    if _is_nomic(model_name):
        model.max_seq_length = NOMIC_MAX_SEQ_LENGTH

    return model


# Helper function to ensure that there are at least 2 texts in the group
def _require_group(texts):
    if len(texts) < 2:
        raise ValueError("requires at least 2 texts")


# Helper function to compute Jaccard distance between two sets
def jaccard_distance(a, b):
    """1 - |A n B| / |A u B|, with the union floored at 1 to avoid division by zero."""
    return 1 - len(a & b) / max(len(a | b), 1)


def semantic_diversity(texts, model_name=DEFAULT_SBERT_MODEL, model=None):
    """D_sem for one group of texts: mean pairwise cosine distance of text embeddings"""

    # Import cosine_similarity from sklearn.metrics.pairwise
    from sklearn.metrics.pairwise import cosine_similarity

    # At least two texts in a group
    _require_group(texts)

    # Load embedding model (either from provided model or by name)
    model = model if model is not None else _get_sbert_model(model_name)

    # Nomic Embed models require a task prefix on every input string
    enc_texts = list(texts)
    if _is_nomic(model_name):
        enc_texts = [NOMIC_TASK_PREFIX + t for t in enc_texts]

    # Embed each full text (not sentence-by-sentence); small batch keeps peak
    # memory bounded on long texts (see pairwise.semantic_distance_matrix)
    embs = model.encode(enc_texts, show_progress_bar=False, normalize_embeddings=True,
                        batch_size=8)

    # Compute pairwise cosine similarity matrix for the embeddings
    sim_matrix = cosine_similarity(embs)

    # Compute mean pairwise cosine distance (1 - cosine similarity) across all pairs of texts
    pairs = list(combinations(range(len(texts)), 2))
    dists = [1 - sim_matrix[i, j] for i, j in pairs]

    return float(np.mean(dists))


# Helper function to compute the set of word unigrams for a single text
def word_unigrams(text):
    return set(word_tokenize_simple(text))


# # Kept for callers that reconstruct the metric inline (not used by lexical_diversity)
# def word_bigrams(text):
#     tokens = word_tokenize_simple(text)

#     return set(zip(tokens[:-1], tokens[1:]))


def lexical_diversity(texts):
    """D_lex for one group of texts: mean pairwise Jaccard distance of word unigrams."""

    # At least two texts in a group
    _require_group(texts)

    # Compute the word-unigram set for each text in the group
    unigram_sets = [word_unigrams(t) for t in texts]

    # Skip pairs where both texts are empty (nothing to compare)
    pairs = list(combinations(range(len(texts)), 2))
    valid = [(i, j) for i, j in pairs if unigram_sets[i] or unigram_sets[j]]

    # If no valid pairs exist (i.e., every text is empty), raise an error
    if not valid:
        raise ValueError("no text in the group has >= 1 token; cannot compute word unigrams")

    # Compute mean pairwise Jaccard distance across valid pairs of texts
    dists = [jaccard_distance(unigram_sets[i], unigram_sets[j]) for i, j in valid]

    return float(np.mean(dists))


def pos_bigrams(text, nlp=None):
    """Set of POS-tag bigrams for one text, via spaCy. Empty if text has < 2 tokens."""

    # SpaCy NLP model (en_core_web_sm) is required for POS tagging
    nlp = nlp if nlp is not None else get_spacy_nlp()

    # Check if spaCy is available
    if nlp is None:
        raise RuntimeError("pos_bigrams requires spaCy (en_core_web_sm) to be installed")

    # Compute POS tags for the text and extract bigrams
    tags = [token.pos_ for token in nlp(text)]

    # Return an empty set if there are fewer than 2 tokens (no bigrams possible)
    if len(tags) < 2:
        return set()

    # Return the set of POS-tag bigrams as tuples of (tag1, tag2)
    return set(zip(tags[:-1], tags[1:]))


# Kept for callers that reconstruct the metric inline
# def pos_trigrams(text, nlp=None):
#     """Set of POS-tag trigrams for one text, via spaCy. Empty if text has < 3 tokens."""

#     nlp = nlp if nlp is not None else get_spacy_nlp()

#     if nlp is None:
#         raise RuntimeError("pos_trigrams requires spaCy (en_core_web_sm) to be installed")

#     tags = [token.pos_ for token in nlp(text)]

#     if len(tags) < 3:
#         return set()

#     return set(zip(tags[:-2], tags[1:-1], tags[2:]))


def syntactic_diversity(texts, nlp=None):
    """D_syn for one group of texts: mean pairwise Jaccard distance of POS-tag bigrams"""

    # At least two texts in a group
    _require_group(texts)

    # Compute POS-tag bigrams for each text in the group using spaCy
    nlp = nlp if nlp is not None else get_spacy_nlp()
    bigram_sets = [pos_bigrams(t, nlp=nlp) for t in texts]

    # Skip pairs where both texts are too short to have any bigram (nothing to compare)
    pairs = list(combinations(range(len(texts)), 2))
    valid = [(i, j) for i, j in pairs if bigram_sets[i] or bigram_sets[j]]

    # If no valid pairs exist (i.e., all texts have fewer than 2 tokens), raise an error
    if not valid:
        raise ValueError("no text in the group has >= 2 tokens; cannot compute POS bigrams")

    # Compute mean pairwise Jaccard distance across valid pairs of texts
    dists = [jaccard_distance(bigram_sets[i], bigram_sets[j]) for i, j in valid]

    return float(np.mean(dists))
