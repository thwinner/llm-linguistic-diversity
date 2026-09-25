"""Single source for loading + cleaning the generation datasets.

Every metrics notebook (``metrics/01`` .. ``metrics/07``) and the descriptive-analysis
notebooks should obtain their texts through :func:`load_texts` instead of re-implementing the
JSONL parsing, quality filter and human downsampling. The prose rationale and the exclusion
tables live in ``data_description/descriptive_analysis_{storytelling,translation}.ipynb``.

Pipeline per domain:

1. Load the model generations (4 open-weight LLMs x 200 prompts x 5 samples) and every human
   reference for those prompts. (The Google-Translate baseline in the translation domain is
   not loaded here as the diversity/quality metrics do not use it)

2. Quality filter:

   * model rows: drop ``finish_reason == 'length'`` (truncated) or ``language_ok is False``
     (non-target-script leakage). ``language_ok`` is taken from the generation log, not
     recomputed.
   * human rows: drop language drift and exact-duplicate texts (compared within the
     domain's human references, first occurrence kept).

3. Human downsampling: for domains listed in :data:`HUMAN_CAP`, keep a fixed number of human
   references per prompt, drawn with ``np.random.default_rng(HUMAN_SAMPLE_SEED)`` so every
   group matches the LLM group size. ``story_id`` keeps the reference's *original* position
   in the list, so per-text metric caches keyed by ``(..., story_id)`` stay valid.

Returned frame -> one row per text::

    domain          dataset key ('storytelling_n200', 'translation_ref3', ...)
    prompt_id       int, 0 .. n_prompts-1 (positional -- what notebooks 01-05 key on)
    raw_prompt_id   the dataset's own id ('wp_0007', ...) -- what notebook 06's caches key on
    source          'human' | 'gemma' | 'llama' | 'mistral' | 'qwen'
    story_id    int -- model: the log's sample_idx; human: original index in the reference list
    prompt      str -- prompt / source paragraph
    text        str -- story / translation
    model       str -- display label ('Human', 'Gemma-2-9B', ...)
"""

from __future__ import annotations
import hashlib
import json
import re
import numpy as np
import pandas as pd
from common.constants import REPO_ROOT, MODEL_KEYS, MODEL_LABELS

__all__ = [
    "REPO_ROOT", "GENERATIONS_DIR", "N_PROMPTS",
    "DOMAIN_CONFIG", "HUMAN_CAP", "HUMAN_SAMPLE_SEED",
    "load_texts", "load_many", "to_groups", "selected_human_ids", "exclusion_summary",
    "dataframe_fingerprint",
]

GENERATIONS_DIR = REPO_ROOT / "data_generation" / "generations"


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
N_PROMPTS = 200

# Non-target-script leakage: characters that should not appear in a clean output
# Storytelling target is English, translation target German -> both Latin script 
# CJK Unified + Hiragana/Katakana + Hangul catch the storytelling drift
# translation additionally flags Cyrillic because some Par3 sources are Russian

_CJK = "一-鿿぀-ヿ가-힯"  # CJK Unified, Hiragana/Katakana, Hangul
_CYRILLIC = "Ѐ-ӿ"
_LANG_DRIFT_RE = {
    "storytelling_n200": re.compile(f"[{_CJK}]"),
    "translation_ref3": re.compile(f"[{_CJK}{_CYRILLIC}]"),
    "translation_ref4": re.compile(f"[{_CJK}{_CYRILLIC}]"),
}


# --------------------------------------------------------------------------------------------
# Loading / cleaning
# --------------------------------------------------------------------------------------------

# The repo root is the first parent directory that contains both the metrics library and the
# data_generation subdirectory. This is used to locate the generations data and the prompts

DOMAIN_CONFIG = {
    "storytelling_n200": dict(
        data_dir="storytelling_n200",
        gen_file=lambda key: f"writingprompts_{key}_n5.jsonl",
        prompts_file="prompts_sample.jsonl",
        text_field="story",
        human_field="human_stories",
        prompt_field="prompt",
    ),
    "translation_ref3": dict(
        data_dir="translation_ref3",
        gen_file=lambda key: f"par3_{key}_n5.jsonl",
        prompts_file="par3_paragraphs_sample.jsonl",
        text_field="translation",
        human_field="human_translations",
        prompt_field="source_text",
    ),
    "translation_ref4": dict( 
        data_dir="translation_ref4",
        gen_file=lambda key: f"par3_{key}_n5.jsonl",
        prompts_file="par3_paragraphs_sample.jsonl",
        text_field="translation",
        human_field="human_translations",
        prompt_field="source_text",
    ),
}

# Human references are downsampled to this many per prompt so every group has the LLM group
# size (5 samples/prompt)

HUMAN_CAP = {"storytelling_n200": 5}
HUMAN_SAMPLE_SEED = 20260902

_DROP_COLS = ["_drop", "_reason"]


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------
def _load_raw(domain: str, n_prompts: int) -> pd.DataFrame:
    '''Load one domain, no cleaning or downsampling. 
    Returns a frame with bookkeeping columns _drop / _reason for the quality filter and human cap.'''

    cfg = DOMAIN_CONFIG[domain]
    ddir = GENERATIONS_DIR / cfg["data_dir"]

    # Load the prompts and human references
    prompt_ids: list = []
    prompt_text: dict = {}
    human_raw: dict = {}
    with open(ddir / cfg["prompts_file"]) as f:
        for line in f:
            r = json.loads(line)
            pid = r["prompt_id"]
            prompt_ids.append(pid)
            prompt_text[pid] = r[cfg["prompt_field"]]
            human_raw[pid] = r[cfg["human_field"]]

    # Keep only the first n_prompts prompts (in file order) and build a prompt_id -> index mapping
    prompt_ids = prompt_ids[:n_prompts]
    pidx = {pid: i for i, pid in enumerate(prompt_ids)}

    rows: list = []

    # Human references: story_id keeps the original position in the reference list
    for pid, i in pidx.items():
        val = human_raw[pid]

        # The human field is a list of strings for storytelling, a dict of {annotator: string} for translation
        texts = list(val.values()) if isinstance(val, dict) else list(val)

        # Add one row per human reference, with the original story_id (index in the list)
        for sid, text in enumerate(texts):
            rows.append(dict(
                domain=domain, prompt_id=i, raw_prompt_id=pid, source="human", story_id=sid,
                prompt=prompt_text[pid], text=text, _drop=False, _reason="",
            ))

    # Model generations: story_id is the log's sample_idx (matches existing per-text caches)
    for key in MODEL_KEYS:
        with open(ddir / cfg["gen_file"](key)) as f:
            for line in f:
                r = json.loads(line)
                pid = r["prompt_id"]

                # Skip any prompt_id that was not in the first n_prompts (the generation logs are for all prompts)
                if pid not in pidx:
                    continue

                # Quality filter: drop truncated or non-target-script rows, keep the bookkeeping columns
                bad_len = r.get("finish_reason") == "length"
                bad_lang = not r.get("language_ok", True)
                reason = "truncated" if bad_len else ("language_drift" if bad_lang else "")

                rows.append(dict(
                    domain=domain, prompt_id=pidx[pid], raw_prompt_id=pid, source=key,
                    story_id=int(r["sample_idx"]),
                    prompt=prompt_text[pid], text=r[cfg["text_field"]],
                    _drop=bool(bad_len or bad_lang), _reason=reason,
                ))

    return pd.DataFrame(rows)


def _flag_human(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Flag human rows with non-target-script drift or a duplicated text (first kept)."""
    drift_re = _LANG_DRIFT_RE[domain]
    human = df.index[df["source"].eq("human")]

    # Flag human rows with non-target-script drift
    drift = df.loc[human, "text"].apply(lambda s: bool(drift_re.search(s)))
    drift_idx = drift.index[drift]
    df.loc[drift_idx, "_drop"] = True
    df.loc[drift_idx, "_reason"] = "language_drift"

    # Flag exact-duplicate human rows (first occurrence kept)
    dup = df.loc[human, "text"].duplicated(keep="first")
    dup_idx = [i for i in dup.index[dup] if i not in set(drift_idx)]
    df.loc[dup_idx, "_drop"] = True
    df.loc[dup_idx, "_reason"] = "exact_duplicate"
    return df


def _cap_human(df: pd.DataFrame, domain: str, seed: int) -> pd.DataFrame:
    """Keep at most ``HUMAN_CAP[domain]`` clean human references per prompt (seeded draw)."""
    cap = HUMAN_CAP.get(domain)
    if cap is None:
        return df

    rng = np.random.default_rng(seed)

    # Keep only clean human references
    clean_human = df[df["source"].eq("human") & ~df["_drop"]]

    # For each prompt, if there are more than cap clean human references, randomly select cap to keep
    for _pid, g in clean_human.groupby("prompt_id", sort=True):
        sids = np.sort(g["story_id"].to_numpy())
        if len(sids) <= cap:
            continue

        # Randomly select cap story_ids to keep, mark the rest as dropped with reason "human_cap"
        keep = set(rng.choice(sids, size=cap, replace=False).tolist())
        drop_idx = g.index[~g["story_id"].isin(keep)]

        df.loc[drop_idx, "_drop"] = True
        df.loc[drop_idx, "_reason"] = "human_cap"

    return df


def load_texts(
    domain: str,
    n_prompts: int = N_PROMPTS,
    *,
    human_seed: int = HUMAN_SAMPLE_SEED,
    apply_human_cap: bool = True,
    with_dropped: bool = False,
) -> pd.DataFrame:
    """Load one domain, cleaned and (where configured) human-downsampled.

    Parameters:
    domain: One of :data:`DOMAIN_CONFIG` (``'storytelling_n200'``, ``'translation_ref3'``, ...).
    n_prompts: Keep only the first ``n_prompts`` prompts (in file order).
    human_seed: Seed for the per-prompt human downsampling draw.
    apply_human_cap: If ``True`` (default) downsample human references to :data:`HUMAN_CAP` per prompt.
        Set ``False`` for the descriptive notebooks, which describe the full cleaned
        reference corpus and document the cap separately.
    with_dropped: If ``True`` keep every row and return the bookkeeping columns ``_drop`` / ``_reason``
        If ``False`` (default) filtered rows are removed and the bookkeeping columns dropped.
    """
    if domain not in DOMAIN_CONFIG:
        raise KeyError(f"unknown domain {domain!r}; known: {list(DOMAIN_CONFIG)}")

    # Load the raw data and 
    df = _load_raw(domain, n_prompts)
    df = _flag_human(df, domain)
    if apply_human_cap:
        df = _cap_human(df, domain, human_seed)
    df["model"] = df["source"].map(MODEL_LABELS)

    # Return only the requested columns, optionally keeping the bookkeeping columns for the descriptive notebooks
    cols = ["domain", "prompt_id", "raw_prompt_id", "source", "story_id", "prompt", "text", "model"]
    if with_dropped:
        return df[cols + _DROP_COLS].reset_index(drop=True)
    return df.loc[~df["_drop"], cols].reset_index(drop=True)


def load_many(domains, **kwargs) -> pd.DataFrame:
    """:func:`load_texts` concatenated over several domains."""
    return pd.concat([load_texts(d, **kwargs) for d in domains], ignore_index=True)


def dataframe_fingerprint(frame: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Content hash of ``cols`` over ``frame``'s rows (in row order), truncated to 16 hex chars.

    Use as a cache-filename (or cache-payload) suffix in every notebook that caches per-text or
    per-group results keyed by identity columns (prompt_id/source/story_id or a notebook's own
    renamed equivalents): any change to the loaded corpus (quality filter, human cap, human
    sample seed or a metric's own preprocessing) changes the fingerprint, so a stale cache
    can never be silently reused under an unchanged row/group key.

    ``cols`` defaults to :func:`load_texts`'s own output columns; pass an explicit list for a
    notebook that renamed or dropped columns after loading (translation notebooks rename
    ``text`` to ``translation``, storytelling ones to ``story``, etc.).
    """

    if cols is None:
        cols = ["domain", "prompt_id", "source", "story_id", "text"]

    # Compute a SHA256 hash of the concatenated string representation of each row in the specified columns
    h = hashlib.sha256()
    for row in frame[cols].itertuples(index=False, name=None):
        h.update(("\x1f".join(map(str, row)) + "\n").encode("utf-8"))

    return h.hexdigest()[:16]


def to_groups(df: pd.DataFrame, min_size: int = 2) -> dict:
    """``(domain, prompt_id, source) -> [texts]`` for the group-level diversity metrics.

    Groups with fewer than ``min_size`` texts are dropped (a group needs >= 2 texts for a
    pairwise diversity score).
    """
    groups: dict = {}
    for key, g in df.groupby(["domain", "prompt_id", "source"], sort=False):
        if len(g) >= min_size:
            groups[key] = g["text"].tolist()
    return groups


def selected_human_ids(domain: str, n_prompts: int = N_PROMPTS, *, human_seed: int = HUMAN_SAMPLE_SEED) -> dict:
    """``prompt_id -> sorted list of kept human story_ids`` after the cap (provenance)."""

    df = load_texts(domain, n_prompts, human_seed=human_seed)
    human = df[df["source"].eq("human")]

    # Only keep prompts that have at least one human reference after the cap (some prompts have no clean human references)
    return {pid: sorted(g["story_id"].tolist()) for pid, g in human.groupby("prompt_id")}


def exclusion_summary(
    domain: str, n_prompts: int = N_PROMPTS, *,
    human_seed: int = HUMAN_SAMPLE_SEED, apply_human_cap: bool = True,
) -> pd.DataFrame:
    """Count of dropped rows by ``source`` x ``_reason``

    Pass ``apply_human_cap=False`` to see only the quality drops (drift / duplicates /
    truncation), without the ``human_cap`` rows.
    """
    df = load_texts(domain, n_prompts, human_seed=human_seed,
                    apply_human_cap=apply_human_cap, with_dropped=True)
    
    dropped = df[df["_drop"]]
    if dropped.empty:
        return pd.DataFrame(columns=["source", "_reason", "n"])

    # Count the number of dropped rows by source and reason
    return (
        dropped.groupby(["source", "_reason"]).size().rename("n").reset_index()
        .sort_values(["source", "_reason"]).reset_index(drop=True)
    )


def summarise() -> None:
    """Print a per-domain load summary"""
    for domain in DOMAIN_CONFIG:
        try:
            df = load_texts(domain)
        except FileNotFoundError as exc:
            print(f"{domain}: skipped ({exc})")
            continue

        # Group size summary (texts per prompt x source)
        sizes = df.groupby(["prompt_id", "source"]).size()
        print(f"\n=== {domain} ===")
        print(f"{len(df):,} texts | "
              f"{df['source'].eq('human').sum():,} human, "
              f"{(~df['source'].eq('human')).sum():,} model")
        print("group size (texts per prompt x source): "
              f"min={sizes.min()} median={int(sizes.median())} max={sizes.max()}")

        # Exclusion summary (dropped rows by source x reason)
        excl = exclusion_summary(domain)
        if not excl.empty:
            print(excl.to_string(index=False))
