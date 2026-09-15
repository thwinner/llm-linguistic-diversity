"""
Shared constants for the thesis analysis notebooks

The model key/label/colour/order definitions and `find_repo_root` are stored here so that they can be 
imported by the notebooks without having to copy-paste them into each notebook.
"""

from __future__ import annotations
from pathlib import Path


def find_repo_root(start: Path | str | None = None) -> Path:
    """Nearest ancestor containing ``metrics/metrics_lib/`` and ``data_generation/generations/``."""

    start = Path(start or Path.cwd()).resolve()

    # Walk up the directory tree until we find a directory that contains both the metrics and data_generation subdirectories
    for p in [start, *start.parents]:
        if (p / "metrics" / "metrics_lib").is_dir() and (p / "data_generation" / "generations").is_dir():
            return p

    # If we reach here, we didn't find the repo root
    raise FileNotFoundError(
        "Could not locate the repo root (expected 'metrics/metrics_lib/' and "
        "'data_generation/generations/')."
    )


REPO_ROOT = find_repo_root()

# The four open-weight instruction-tuned LLMs compared against the human references
MODEL_KEYS = ["gemma", "llama", "mistral", "qwen"]

# key -> Hugging Face model id
MODELS = {
    "gemma":   "google/gemma-2-9b-it",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
}

# source key -> short display label
MODEL_LABELS = {
    "gemma": "Gemma-2-9B",
    "llama": "Llama-3.1-8B",
    "mistral": "Mistral-7B-v0.3",
    "qwen": "Qwen2.5-7B",
    "human": "Human",
}

# Google-Translate baseline: translation domain only
# The translation notebook opts in with MODEL_LABELS = {**MODEL_LABELS, GT_KEY: GT_LABEL}

GT_KEY = "gt"
GT_LABEL = "GT (Google Translate)"

# display label -> hex colour
MODEL_COLORS = {
    "Human":                 "#cb1f73",
    "GT (Google Translate)": "#7f7f7f",
    "Gemma-2-9B":            "#fcc72d",
    "Llama-3.1-8B":          "#ea6d3d",
    "Mistral-7B-v0.3":       "#e03a3c",
    "Qwen2.5-7B":            "#383a6b",
}

# Plot / table ordering:
#   - GEN_ORDER = the four LLMs
#   - MODEL_ORDER = GEN_ORDER with Human prepended
#   - MODEL_ORDER_GT = MODEL_ORDER with GT (Google Translate) inserted
GEN_ORDER = ["Gemma-2-9B", "Llama-3.1-8B", "Mistral-7B-v0.3", "Qwen2.5-7B"]
MODEL_ORDER = ["Human", *GEN_ORDER]
MODEL_ORDER_GT = ["Human", "GT (Google Translate)", *GEN_ORDER]

# Par3 / translation source languages
LANG_ORDER = ["German", "Russian", "Chinese"]
LANG_COLORS = {"German": "#4C72B0", "Russian": "#55A868", "Chinese": "#C44E52"}

__all__ = [
    "find_repo_root", "REPO_ROOT",
    "MODEL_KEYS", "MODELS", "MODEL_LABELS", "MODEL_COLORS",
    "GT_KEY", "GT_LABEL",
    "GEN_ORDER", "MODEL_ORDER", "MODEL_ORDER_GT",
    "LANG_ORDER", "LANG_COLORS",
]
