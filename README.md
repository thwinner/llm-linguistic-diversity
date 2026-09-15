# Linguistic Diversity in LLM Outputs: A Multidimensional Evaluation Framework

## Setup

Install the dependencies and the local `metrics_lib` package (editable, so changes in `metrics/metrics_lib/` and `common/` take effect immediately):

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

The datasets used in this project are not included in the repository due to licensing and redistribution restrictions. Please obtain them from their official sources and place them in the corresponding subdirectories under `data/`.

- **PAR3**  
  Karpinska et al. (2022), *PAR3*.  
  [[Official repository](https://github.com/katherinethai/par3?utm_source=chatgpt.com)]

- **LITEVAL-CORPUS**  
  Zhang, Zhao, and Eger (2025), *How Good Are LLMs for Literary Translation, Really? Literary Translation Evaluation with Humans and LLMs*.  
  [[Official repository](https://github.com/zhangr2021/LitMT_eval?utm_source=chatgpt.com)] · [[Paper](https://aclanthology.org/2025.naacl-long.548/?utm_source=chatgpt.com)]

- **Human Drift**  
  [[Official repository](https://github.com/EstebanGarces/human_drift?utm_source=chatgpt.com)] · [[Paper](https://openreview.net/pdf?id=OzhrGQvJpO&utm_source=chatgpt.com)]

- **HANNA**  
  Chhun et al. (2022), *Of Human Criteria and Automatic Metrics: A Benchmark of the Evaluation of Story Generation*.  
  [[Official repository](https://github.com/dig-team/hanna-benchmark-asg?utm_source=chatgpt.com)] · [[Paper](https://aclanthology.org/2022.coling-1.509/?utm_source=chatgpt.com)]

The expected directory structure is:

```text
data/
├── human_drift/
├── liteval_corpus/
├── LITEVAL-CORPUS/
├── par3/
└── hanna_stories_annotations.csv
```

## Data Generation

The generation pipeline is organized as follows:

```text
data_generation/
├── generations/
│   ├── storytelling_n200/
│   ├── translation_ref3/
│   └── translation_ref4/
├── par3_generation.ipynb
└── writingprompts_generation.ipynb
```

The notebooks in `data_generation/` contain the generation pipelines for the translation and storytelling experiments.

- `par3_generation.ipynb` is used to generate translations for the PAR3 experiments.  
- `writingprompts_generation.ipynb` is used to generate stories for the WritingPrompts experiments.  
- `storytelling_n200/` contains the generated stories used in the storytelling experiments.
- `translation_ref3/` and `translation_ref4/` contain the generated translations for the corresponding translation settings.

The generated outputs in `data_generation/generations/` are not included in the repository because some files also contain material from the original datasets.