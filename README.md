# Masterthesis


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

data/
├── human_drift/
├── liteval_corpus/
├── LITEVAL-CORPUS/
├── par3/
└── hanna_stories_annotations.csv

Generated model outputs are stored locally in `data_generation/generations/` and are not distributed with this repository. 