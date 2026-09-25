# Linguistic Diversity in LLM Outputs: A Multidimensional Evaluation Framework

Code and analysis pipeline for the Master's thesis *Linguistic Diversity in LLM Outputs: A Multidimensional Evaluation Framework*.

## Overview

This repository investigates linguistic diversity in repeatedly sampled LLM outputs and compares it with variation among human texts for the same input. The analysis covers two tasks, open-ended storytelling and literary translation, and evaluates diversity across six complementary dimensions.

| Metric | Dimension | Representation | Distance |
|---|---|---|---|
| `D_sem` | semantic | Nomic Embed Text v1.5 embeddings | cosine |
| `D_lex` | lexical | Word unigrams | Jaccard |
| `D_syn` | syntactic | POS-tag bigrams | Jaccard |
| `D_disc` | discourse | Entity grids | Jensen–Shannon |
| `D_narr` | narrative | 10-segment sentiment arcs | Euclidean |
| `D_style` | stylistic | Stylometric features and function-word profiles | Euclidean / cosine |

For each dimension, within-source diversity is computed from pairwise distances among texts generated for the same input. Additional analyses compare model outputs with human reference texts and examine whether distance from human writing is associated with human quality judgments. The repository also includes four single-text baseline metrics following Garces Arias et al. (2025): Diversity, Perplexity, Coherence and Q*Text.

## Experimental Setup

The experiments cover two generation tasks, namely storytelling and translation.

| Domain | Task | Source data | Human references |
|---|---|---|---|
| `storytelling_n200` | Open-ended story continuation | WritingPrompts (Fan et al., 2018), test split | 5 per prompt, sampled with a fixed seed from 5–45 available references |
| `translation_ref3` | Literary translation into German | PAR3 (Karpinska et al., 2022) | 3 translations per source paragraph |
| `translation_ref4` | Literary translation into German | PAR3 (Karpinska et al., 2022) | 4 translations per source paragraph |

Each domain contains 200 source inputs. Five outputs are generated per input and model.

The following open-weight instruction-tuned models are evaluated:

- Gemma 2 9B Instruct (`google/gemma-2-9b-it`; Gemma Team et al., 2024)
- Llama 3.1 8B Instruct (`meta-llama/Llama-3.1-8B-Instruct`; Grattafiori et al., 2024)
- Mistral 7B Instruct v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`; Jiang et al., 2023)
- Qwen2.5 7B Instruct (`Qwen/Qwen2.5-7B-Instruct`; Qwen Team, 2024)

All models are run locally with vLLM (Kwon et al., 2023) using the same decoding configuration: `temperature = 1.0`, `top_p = 0.9`, `repetition_penalty = 1.0` and a fixed random seed of `42`. Five samples are generated per input and model. The maximum generation length is 1,280 new tokens for storytelling and 3,072 new tokens for translation, with a maximum model length of 8,192 tokens for translation. A Google Translate baseline is additionally included for the translation task.

Human quality analyses use two externally annotated datasets:
- **HANNA** (Chhun et al., 2022) for story-quality ratings
- **LITEVAL-CORPUS** (Zhang, Zhao, & Eger, 2025) for MQM-based literary translation quality


## Repository Structure

```text
common/
    shared constants, model labels, colours, repository paths and data_prep.py for loading and cleaning data

data/
    external datasets required for the analyses

data_generation/
    generation notebooks and generated model outputs

data_description/
    descriptive analyses for the individual datasets

metrics/
├── metrics_lib/
│   Reusable metric implementations
├── 01_discourse_diversity/
│   Discourse diversity
├── 02_stylistic_diversity/
│   Stylistic diversity
├── 03_narrative_diversity/
│   Narrative diversity
├── 04_baseline_metrics/
│   Diversity, Perplexity, Coherence, and Q*Text
├── 05_sem_lex_syn_diversity/
│   Robustness analyses for semantic, lexical and syntactic diversity
├── 06_writingprompts_diversity/
│   Full diversity analysis for storytelling
└── 07_par3_diversity/
    Full diversity analysis for translation

quality/
    HANNA and LITEVAL quality analyses and result tables

figures/
    figures produced by the analysis notebooks

thesis_figures.ipynb
    reporting-only notebook that reads saved results and creates the figures used in the thesis

prototypes/
    earlier exploratory analyses retained for reference
```

Each metric notebook writes its result tables to its own `tables/` directory and its plots to the corresponding `figures/` directory.

`thesis_figures.ipynb` does not recompute the analyses. It reads the saved result files and renders the final figures into `Masterthesis_Latex/figures/`.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is required

No further installation step is needed. The notebooks add the repository root to `sys.path` themselves, so `metrics/metrics_lib/` and `common/` are importable as soon as the repository is checked out, and changes to them take effect immediately.


## Data Availability

The external datasets used in this project are not included in the repository because of licensing and redistribution restrictions. They must be obtained from their official sources and placed in the corresponding subdirectories under `data/`. External datasets and model weights remain subject to their original licenses and terms of use.

### PAR3

Karpinska et al. (2022), *PAR3*.

- [Official repository](https://github.com/katherinethai/par3)

### LITEVAL-CORPUS

Zhang, Zhao, and Eger (2025), *How Good Are LLMs for Literary Translation, Really? Literary Translation Evaluation with Humans and LLMs*.

- [Official repository](https://github.com/zhangr2021/LitMT_eval)
- [Paper](https://aclanthology.org/2025.naacl-long.548/)

### Human Drift

- [Official repository](https://github.com/EstebanGarces/human_drift)
- [Paper](https://openreview.net/pdf?id=OzhrGQvJpO)

### HANNA

Chhun et al. (2022), *Of Human Criteria and Automatic Metrics: A Benchmark of the Evaluation of Story Generation*.

- [Official repository](https://github.com/dig-team/hanna-benchmark-asg)
- [Paper](https://aclanthology.org/2022.coling-1.509/)

The expected directory structure is:

```text
data/
├── human_drift/
├── liteval_corpus/
├── LITEVAL-CORPUS/
├── par3/
└── hanna_stories_annotations.csv
```

The generated outputs under `data_generation/generations/` are also not tracked because some files contain material from the original datasets.

`data_generation/generations_public/` contains an ID-only version produced by `make_generations_public.py`. It retains identifiers and model generations while removing copyright-protected prompts, source texts and human references.

## Data Generation

The generation notebooks are located in `data_generation/`:

```text
data_generation/
├── generations/
│   ├── storytelling_n200/
│   ├── translation_ref3/
│   └── translation_ref4/
├── par3_generation.ipynb
└── writingprompts_generation.ipynb
```

`writingprompts_generation.ipynb` generates the story continuations used in the storytelling experiments.
`par3_generation.ipynb` generates the model translations used in the PAR3 experiments.

The task instructions and prompt templates used for generation are defined directly in the corresponding generation notebooks.

Generation requires a GPU. When using vLLM, models should be run in separate sessions because GPU memory is not reliably released when switching models within the same session.


## Loading the Data

All analysis notebooks load and preprocess texts through a shared function:

```python
from common.data_prep import load_texts

df = load_texts("storytelling_n200")
```

The loader provides a common preprocessing pipeline across analyses. It:

- Removes truncated model outputs with `finish_reason == "length"`
- Removes model outputs with non-target-script leakage
- Removes language drift from human references
- Removes exact duplicate human references
- Downsamples human references to a fixed number per input using a fixed seed

This ensures that the same cleaning and sampling procedure is applied across the metric and quality analyses.

A summary of the available domains can be generated with:

```bash
python -m common
```

## Using the Metrics Library

The group-level diversity metrics take a list of texts belonging to the same group and return one diversity score.

```python
from metrics.metrics_lib import (
    semantic_diversity,
    lexical_diversity,
    syntactic_diversity,
    discourse_diversity,
    narrative_diversity,
)

texts = [...]
d_sem = semantic_diversity(texts)
```

Two metrics require dataset-level information rather than a single group.
- `stylistic_diversity()` receives the full set of groups because its components are z-standardized and normalized across the groups being compared.
- `compute_qstar()` likewise operates on dataset-relative arrays rather than individual scalar values


### Pairwise Distances

The functions in `metrics_lib.pairwise` provide access to the pairwise distance matrices before distances are aggregated into group-level diversity scores.

```python
from metrics.metrics_lib.pairwise import (
    distance_matrix,
    within_between,
    slices_from_lengths,
)

D = distance_matrix(
    "D_sem",
    human_texts + gemma_texts,
)

wb = within_between(
    D,
    slices_from_lengths([
        ("human", len(human_texts)),
        ("gemma", len(gemma_texts)),
    ]),
)

wb.within["gemma"]
wb.between[("human", "gemma")]
```

These pairwise distances are used for the within-source diversity analyses and the comparisons between model outputs and human references.


## Reproducing the Analysis

To reproduce the complete analysis:

1. Obtain the external datasets and place them under `data/` as described above.
2. Generate the model outputs with:
   - `data_generation/writingprompts_generation.ipynb`
   - `data_generation/par3_generation.ipynb`
3. Run the metric notebooks under `metrics/01_*` through `metrics/07_*`.
4. Run the quality analyses under `quality/`.
5. Run `thesis_figures.ipynb` to generate the final thesis figures.

The metric notebooks write their results to local `tables/` directories. Several notebooks additionally cache expensive per-text computations in `cache/`.

Notebooks `01`–`04` can be switched between the two main domains using the `DOMAIN` variable at the beginning of each notebook:

```python
DOMAIN = "storytelling_n200"
```
or
```python
DOMAIN = "translation_ref3"
```

## References
- Barzilay, R., & Lapata, M. (2008). *Modeling Local Coherence: An Entity-Based Approach.*
- Chhun, C., Colombo, P., Clavel, C., & Bellot, P. (2022). *Of Human Criteria and Automatic Metrics: A Benchmark of the Evaluation of Story Generation.*
- Fan, A., Lewis, M., & Dauphin, Y. (2018). *Hierarchical Neural Story Generation.*
- Garces Arias, E. et al. (2025). *Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework.*
- Gemma Team et al. (2024). *Gemma 2: Improving Open Language Models at a Practical Size.*
- Grattafiori et al. (2024). *The Llama 3 Herd of Models.*
- Jiang et al. (2023). *Mistral 7B.*
- Karpinska, M. et al. (2022). *DEMETR / PAR3.*
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.*
- Qwen Team (2024). *Qwen2.5 Technical Report.*
- Reagan, A. J. et al. (2016). *The Emotional Arcs of Stories Are Dominated by Six Basic Shapes.*
- Zhang, R., Zhao, W., & Eger, S. (2025). *How Good Are LLMs for Literary Translation, Really? Literary Translation Evaluation with Humans and LLMs.*