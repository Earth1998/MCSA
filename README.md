# MCSA

<div align="center">
  
  [![GitHub stars](https://badgen.net/github/stars/Earth1998/MCSA?_t=1681000000)](https://GitHub.com/Earth1998/MCSA/stargazers/)
  [![GitHub forks](https://badgen.net/github/forks/Earth1998/MCSA/?_t=1681000000)](https://GitHub.com/Earth1998/MCSA/network/)
  [![GitHub issues](https://badgen.net/github/issues/Earth1998/MCSA/?color=red)](https://GitHub.com/Earth1998/MCSA/issues/)
  [![GitHub license](https://img.shields.io/github/license/Earth1998/MCSA.svg?_t=1681234567)](https://github.com/Earth1998/MCSA/blob/master/LICENSE)

</div>

Predicting drug response and understanding the corresponding pharmacogenomic biology is crucial for precision oncology, drug discovery, and clinical strategy. However, data heterogeneity in dynamic environments is ubiquitous in practice, with data distributions for drugs, cancer, or institutions continually changing over time, and historical data may become inaccessible due to privacy restrictions. Such changing dynamics undermine the reliability of drug response prediction and interpretation, severely misleading the exploration of cancer treatment mechanisms. Here, we propose a novel **interpretable multi-contextual self-alignment framework (MCSA)** that enables continual learning and alignment of drug response knowledge across multiple contexts, effectively mining pharmacogenomic biomarkers from dynamic task streams. MCSA initially performs continual adversarial self-supervised learning to achieve local alignment, transferring drug response knowledge from the representation context of previous tasks into the current representation. It then leverages interpretability-consistency regularization to perform interpretability context alignment, guiding the learning of the model and the plug-in pharmacogenomic interpretable module (PIPGIM) to continually explore pharmacogenomic biology. Finally, using cross-model interactive learning, MCSA further mitigates mutual interference between tasks in the global alignment of model contexts, improving the framework performance and knowledge diversity. Experiments on multiple continual learning scenarios have shown that MCSA outperforms state-of-the-art baseline methods and effectively alleviates catastrophic forgetting and interpretable concept drift In addition, MCSA has performed well in drug response biomarker discovery, clinical chemotherapy response prediction, and prognosis analysis, consistent with clinical results, indicating the potential of MCSA in developing personalized therapies and revealing pharmacogenomic biology in changing dynamics.

**Given the complexity and instability of individuals in configuring the environment, we strongly recommend that users use MCSA's online prediction Web server, which can be accessed through **http://bliulab.net/MCSA/**.**

![Model](/imgs/Model.png)

**Fig. 1: The model architecture of MCSA.** MCSA is a continual learning framework that enables drug response prediction and analysis through progressive alignment of local, interpretability, and global contexts in dynamic environments.

# 1 Installation

## 1.1 Create conda environment

```
conda create -n gear python=3.10
conda activate gear
```
For installing conda, please refer to https://docs.anaconda.com/free/miniconda/.

## 1.2 Requirements
The main dependencies used in this project are as follows (for more information, please see the `environment.yaml` file):

```
python  3.10
biopython 1.84
huggingface-hub 0.26.1
numpy 2.1.2
transformers 4.46.0
tokenizers 0.20.1
sentencepiece 0.2.0
torch 2.5.0+cpu
torchaudio 2.5.0+cpu
torchvision 0.20.0+cpu
torch-geometric 2.6.1
```

> **Note** If you have an available GPU, the accelerated MCSA can be used to predict peptide-protein binary interactions and pair-specific binding residues. Change the URL below to reflect your version of the cuda toolkit (cu118 for cuda=11.6 and cuda 11.8, cu121 for cuda 12.1). However, do not provide a number greater than your installed cuda toolkit version!
> 
> ```
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
> ```
>
> For more information on other cuda versions, see the [pytorch installation documentation](https://pytorch.org/).

## 1.3 Tools
Feature extraction tools and databases on which KEIPA relies:

## 1.4 Inatsll MCSA
To install from the development branch run
```
git clone git@github.com:Earth1998/MCSA.git
cd MCSA/
```

**Finally, configure the defalut path of the above tool and the database in `conf.py`. You can change the path of the tool and database by configuring `conf.py` as needed.**

# 2 Usage
It takes 2 steps to predict peptide-protein binary interaction and peptide-protein-specific binding residues:

# 3 Problem feedback
If you have questions on how to use MCSA, feel free to raise questions in the [discussions section](https://github.com/Earth1998/MCSA/discussions). If you identify any potential bugs, feel free to raise them in the [issuetracker](https://github.com/Earth1998/MCSA/issues).

In addition, if you have any further questions about MCSA, please feel free to contact us [**thquan@bliulab.net**]

# 4 Citation

If you find our work useful, please cite us at
```
@article{Quan2025Multi,
  title={Multi-contextual self-alignment framework for interpretable continual learning in predicting drug response and exploring pharmacogenomic biology},
  author={Tianhong Quan, Ke Yan, Shutao Chen, and Bin Liu},
  journal={submitted},
  year={2025},
  publisher={}
}

```
