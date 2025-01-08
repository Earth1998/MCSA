# MCSA

<div align="center">
  
  [![GitHub stars](https://badgen.net/github/stars/Earth1998/MCSA?_t=1681000000)](https://GitHub.com/Earth1998/MCSA/stargazers/)
  [![GitHub forks](https://badgen.net/github/forks/Earth1998/MCSA/?_t=1681000000)](https://GitHub.com/Earth1998/MCSA/network/)
  [![GitHub issues](https://badgen.net/github/issues/Earth1998/MCSA/?color=red)](https://GitHub.com/Earth1998/MCSA/issues/)
  [![GitHub license](https://img.shields.io/github/license/Earth1998/MCSA.svg?_t=1681234567)](https://github.com/Earth1998/MCSA/blob/master/LICENSE)

</div>

Predicting drug response and understanding the corresponding pharmacogenomic biology is crucial for precision oncology, drug discovery, and clinical strategy. However, data heterogeneity in dynamic environments is ubiquitous in practice, with data distributions for drugs, cancer, or institutions continually changing over time, and historical data may become inaccessible due to privacy restrictions. Such changing dynamics undermine the reliability of drug response prediction and interpretation, severely misleading the exploration of cancer treatment mechanisms. Here, we propose a novel interpretable multi-contextual self-alignment framework (MCSA) that enables continual learning and alignment of drug response knowledge across multiple contexts, effectively mining pharmacogenomic biomarkers from dynamic task streams. MCSA initially performs continual adversarial self-supervised learning to achieve local alignment, transferring drug response knowledge from the representation context of previous tasks into the current representation. It then leverages interpretability-consistency regularization to perform interpretability context alignment, guiding the learning of the model and the plug-in pharmacogenomic interpretable module (PIPGIM) to continually explore pharmacogenomic biology. Finally, using cross-model interactive learning, MCSA further mitigates mutual interference between tasks in the global alignment of model contexts, improving the framework performance and knowledge diversity. Experiments on multiple continual learning scenarios have shown that MCSA outperforms state-of-the-art baseline methods and effectively alleviates catastrophic forgetting and interpretable concept drift In addition, MCSA has performed well in drug response biomarker discovery, clinical chemotherapy response prediction, and prognosis analysis, consistent with clinical results, indicating the potential of MCSA in developing personalized therapies and revealing pharmacogenomic biology in changing dynamics.

![Model](/imgs/Model.png)
