"""Shared loaded resources used by multiple metric modules."""
from functools import lru_cache


@lru_cache(maxsize=1)
def get_spacy_nlp():
    """Load and cache the shared spaCy English pipeline (en_core_web_sm).

    Returns None if spaCy or the model is not installed, so callers can fall
    back to a heuristic instead of failing hard.
    """
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None
