"""Reusable text-diversity/quality metrics extracted from the metrics/ notebooks.

    from metrics_lib import discourse_diversity, discourse_diversity_simple, narrative_diversity, stylistic_diversity
    from metrics_lib import semantic_diversity, lexical_diversity, syntactic_diversity
    from metrics_lib import diversity_score, compute_perplexity, compute_coherence, compute_qstar

- discourse_diversity(), discourse_diversity_simple(), narrative_diversity(),
semantic_diversity(), lexical_diversity() and syntactic_diversity() each take
one group of texts (e.g. multiple generations for the same prompt) and return
a single score.
- stylistic_diversity() needs the whole batch of groups at once (see its
docstring) because its components are normalized across groups.
- compute_qstar() similarly needs dataset-relative arrays, not single scalars.
- pairwise.distance_matrix() / within_between() expose the per-pair distances
behind the group scores, for cross-source comparisons (within-source diversity
vs. distance to a human reference).

Install once (editable), from the repo root: pip install -e .
"""
from .discourse import discourse_diversity, discourse_diversity_simple
from .narrative import narrative_diversity, get_sentiment_arc
from .stylistic import stylistic_diversity, extract_style_features
from .semantic_lexical_syntactic import semantic_diversity, lexical_diversity, syntactic_diversity
from .baseline import diversity_score, compute_perplexity, compute_coherence, compute_qstar
from .pairwise import (
    distance_matrix, within_between, WithinBetween, slices_from_lengths,
    reference_control_splits, reference_control_splits_rowwise, style_distance_matrices,
)

__all__ = [
    "discourse_diversity",
    "discourse_diversity_simple",
    "narrative_diversity",
    "get_sentiment_arc",
    "stylistic_diversity",
    "extract_style_features",
    "semantic_diversity",
    "lexical_diversity",
    "syntactic_diversity",
    "diversity_score",
    "compute_perplexity",
    "compute_coherence",
    "compute_qstar",
    "distance_matrix",
    "within_between",
    "WithinBetween",
    "slices_from_lengths",
    "reference_control_splits",
    "reference_control_splits_rowwise",
    "style_distance_matrices",
]
