"""
Semantic, lexical and syntactic diversity (D_sem, D_lex, D_syn): 
pairwise text-level diversity across a group of texts (e.g. multiple generations for the same prompt)

Here: applied across whole texts within a group instead of across sentences within one text

- D_sem: mean pairwise cosine distance of SBERT text embeddings
- D_lex: mean pairwise Jaccard distance of word bigrams
- D_syn: mean pairwise Jaccard distance of POS-tag trigrams (spaCy)

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

# Default SBERT model for semantic diversity
DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"


# Cached SBERT model loader to avoid reloading the model multiple times
@lru_cache(maxsize=1)
def _get_sbert_model(model_name=DEFAULT_SBERT_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# Helper function to ensure that there are at least 2 texts in the group
def _require_group(texts):
    if len(texts) < 2:
        raise ValueError("requires at least 2 texts")


# Helper function to compute Jaccard distance between two sets
def jaccard_distance(a, b):
    """1 - |A n B| / |A u B|, with the union floored at 1 to avoid division by zero."""
    return 1 - len(a & b) / max(len(a | b), 1)


def semantic_diversity(texts, model_name=DEFAULT_SBERT_MODEL, model=None):
    """D_sem for one group of texts: mean pairwise cosine distance of SBERT embeddings"""

    # Import cosine_similarity from sklearn.metrics.pairwise
    from sklearn.metrics.pairwise import cosine_similarity

    # At least two texts in a group
    _require_group(texts)

    # Load SBERT model (either from provided model or by name)
    model = model if model is not None else _get_sbert_model(model_name)

    # Embed each full text (not sentence-by-sentence)
    embs = model.encode(list(texts), show_progress_bar=False)

    # Compute pairwise cosine similarity matrix for the embeddings
    sim_matrix = cosine_similarity(embs)

    # Compute mean pairwise cosine distance (1 - cosine similarity) across all pairs of texts
    pairs = list(combinations(range(len(texts)), 2))
    dists = [1 - sim_matrix[i, j] for i, j in pairs]

    return float(np.mean(dists))


# Helper function to compute word bigrams for a single text
def word_bigrams(text):
    tokens = word_tokenize_simple(text)

    return set(zip(tokens[:-1], tokens[1:]))


def lexical_diversity(texts):
    """D_lex for one group of texts: mean pairwise Jaccard distance of word bigrams."""

    # At least two texts in a group
    _require_group(texts)

    # Compute word bigrams for each text in the group
    bigram_sets = [word_bigrams(t) for t in texts]

    # Skip pairs where both texts are too short to have any bigram (nothing to compare)
    pairs = list(combinations(range(len(texts)), 2))
    valid = [(i, j) for i, j in pairs if bigram_sets[i] or bigram_sets[j]]

    # If no valid pairs exist (i.e., all texts have fewer than 2 tokens), raise an error
    if not valid:
        raise ValueError("no text in the group has >= 2 tokens; cannot compute word bigrams")

    # Compute mean pairwise Jaccard distance across valid pairs of texts
    dists = [jaccard_distance(bigram_sets[i], bigram_sets[j]) for i, j in valid]

    return float(np.mean(dists))


def pos_trigrams(text, nlp=None):
    """Set of POS-tag trigrams for one text, via spaCy. Empty if text has < 3 tokens."""

    # SpaCy NLP model (en_core_web_sm) is required for POS tagging
    nlp = nlp if nlp is not None else get_spacy_nlp()

    # Check if spaCy is available
    if nlp is None:
        raise RuntimeError("pos_trigrams requires spaCy (en_core_web_sm) to be installed")

    # Compute POS tags for the text and extract trigrams
    tags = [token.pos_ for token in nlp(text)]

    # Return an empty set if there are fewer than 3 tokens (no trigrams possible)
    if len(tags) < 3:
        return set()

    # Return the set of POS-tag trigrams as tuples of (tag1, tag2, tag3)
    return set(zip(tags[:-2], tags[1:-1], tags[2:]))


def syntactic_diversity(texts, nlp=None):
    """D_syn for one group of texts: mean pairwise Jaccard distance of POS-tag trigrams"""

    # At least two texts in a group
    _require_group(texts)

    # Compute POS-tag trigrams for each text in the group using spaCy
    nlp = nlp if nlp is not None else get_spacy_nlp()
    trigram_sets = [pos_trigrams(t, nlp=nlp) for t in texts]

    # Skip pairs where both texts are too short to have any trigram (nothing to compare)
    pairs = list(combinations(range(len(texts)), 2))
    valid = [(i, j) for i, j in pairs if trigram_sets[i] or trigram_sets[j]]

    # If no valid pairs exist (i.e., all texts have fewer than 3 tokens), raise an error
    if not valid:
        raise ValueError("no text in the group has >= 3 tokens; cannot compute POS trigrams")

    # Compute mean pairwise Jaccard distance across valid pairs of texts
    dists = [jaccard_distance(trigram_sets[i], trigram_sets[j]) for i, j in valid]
    
    return float(np.mean(dists))
