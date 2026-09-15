"""Shared helpers for the analysis notebooks (``metrics/``, ``data_description/``).

Usage from a notebook (after the repo root is on ``sys.path``)::

    from common import load_texts, to_groups, MODEL_COLORS, MODEL_ORDER
    from common import data_prep   # for the full module namespace
"""

from common.constants import (
    REPO_ROOT,
    find_repo_root,
    MODEL_KEYS,
    MODELS,
    MODEL_LABELS,
    MODEL_COLORS,
    GT_KEY,
    GT_LABEL,
    GEN_ORDER,
    MODEL_ORDER,
    MODEL_ORDER_GT,
    LANG_ORDER,
    LANG_COLORS,
)
from common import data_prep
from common.data_prep import (
    load_texts,
    load_many,
    to_groups,
    selected_human_ids,
    exclusion_summary,
    DOMAIN_CONFIG,
    HUMAN_CAP,
    HUMAN_SAMPLE_SEED,
    N_PROMPTS,
    GENERATIONS_DIR,
)

__all__ = [
    "REPO_ROOT", "find_repo_root",
    "MODEL_KEYS", "MODELS", "MODEL_LABELS", "MODEL_COLORS", "GT_KEY", "GT_LABEL",
    "GEN_ORDER", "MODEL_ORDER", "MODEL_ORDER_GT", "LANG_ORDER", "LANG_COLORS",
    "data_prep",
    "load_texts", "load_many", "to_groups", "selected_human_ids", "exclusion_summary",
    "DOMAIN_CONFIG", "HUMAN_CAP", "HUMAN_SAMPLE_SEED", "N_PROMPTS", "GENERATIONS_DIR",
]
