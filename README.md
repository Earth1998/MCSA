# MCSA

<div align="center">
  
  [![GitHub stars](https://badgen.net/github/stars/Earth1998/MCSA?_t=1681000000)](https://GitHub.com/Earth1998/MCSA/stargazers/)
  [![GitHub forks](https://badgen.net/github/forks/Earth1998/MCSA/?_t=1681000000)](https://GitHub.com/Earth1998/MCSA/network/)
  [![GitHub issues](https://badgen.net/github/issues/Earth1998/MCSA/?color=red)](https://GitHub.com/Earth1998/MCSA/issues/)
  [![GitHub license](https://img.shields.io/github/license/Earth1998/MCSA.svg?_t=1681234567)](https://github.com/Earth1998/MCSA/blob/master/LICENSE)

</div>

Code for paper "**MCSA: deciphering drug responses and mechanisms of action via dynamic alignment of heterogeneous pharmacogenomic data**" (Submitted).

## Table of Contents
  - [Abstract](#Abstract)
  - [Approach](#Approach)
  - [Installation](#Installation)
    - [Create conda environment](#Create-conda-environment)
    - [Requirements](#Requirements)
    - [Tools and Datasets](#Tools-and-Datasets)
    - [Inatsll MCSA](#Inatsll-MCSA)
  - [Usage](#Usage)
  - [Feedback](#Feedback)
  - [Citation](#Citation)
  - [Acknowledgement](#Acknowledgement)

## Abstract

Predicting and understanding the heterogeneity of drug responses is crucial for precision oncology. In this study, we introduce a multi-contextual self-alignment (MCSA) framework that integrates semantic knowledge of pharmacogenomic data, facilitates cross-domain interactions across heterogeneous drugs, transcriptomic profiles, and multi-institutional datasets, and enables genomic interpretation of both targeted therapies and chemotherapeutic agents. The model is designed to function effectively in both dynamic and static data environments, exhibiting adaptability to diverse drugs, cancer types, and sequencing technologies. By leveraging genomic information, MCSA elucidates drug mechanisms of action through cellular state analysis and assesses the efficacy of clinical chemotherapy. Its universality and relevance to clinical outcomes have been validated through applications across various methods and clinical datasets. Overall, MCSA offers a unified framework for precision oncology and drug discovery, underscoring its potential to advance the life sciences.

**Given the complexity and instability of individuals in configuring the environment, we strongly recommend that users use MCSA's online function analysis server, which can be accessed through **http://bliulab.net/MCSA/**.**

## Approach

![Model](/imgs/Model.png)

**Fig. 1: The model architecture of MCSA.** MCSA is a continual learning framework that enables drug response prediction and analysis through progressive alignment of local, interpretability, and global contexts in dynamic environments.

## Installation

### Create conda environment

```
conda create -n gear python=3.10
conda activate gear
```
For installing conda, please refer to https://docs.anaconda.com/free/miniconda/.

### Requirements
The main dependencies used in this project are as follows (for more information, please see the `environment.yaml` file):

```
python  3.10
scikit-learn 1.3.0
scipy 1.9.3
numpy 1.24.3
pandas 2.0.3
openxyl 3.1.5
nltk 3.9.1
torch 2.5.0+cpu
torchaudio 2.5.0+cpu
torchvision 0.20.0+cpu
torch-geometric 2.6.1
```

> **Note** If you have an available GPU, the accelerated MCSA can be used to predict drug response and analyze biological mechanism. Change the URL below to reflect your version of the cuda toolkit (cu118 for cuda=11.6 and cuda 11.8, cu121 for cuda 12.1). However, do not provide a number greater than your installed cuda toolkit version!
> 
> ```
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
> ```
>
> For more information on other cuda versions, see the [pytorch installation documentation](https://pytorch.org/).

### Tools and Datasets
In this study, various tools were used for feature extraction and downstream analysis, including [TCGAbiolinks](https://www.bioconductor.org/packages/release/bioc/html/TCGAbiolinks.html), [pubchem](https://pubchem.ncbi.nlm.nih.gov), [DESeq2](https://www.bioconductor.org/packages/release/bioc/html/DESeq2.html), [TIMER](https://cistrome.shinyapps.io/timer), and [HPAanalyze](https://www.bioconductor.org/packages/release/bioc/html/HPAanalyze.html).

The datasets used in this study come from the Genomics of Drug Sensitivity in Cancer ([GDSC](https://www.cancerrxgene.org)), the Cancer Cell Line Encyclopedia ([CCLE](https://depmap.org/portal)), and The Cancer Genome Atlas ([TCGA](https://portal.gdc.cancer.gov)), which are used to construct drug-incremental learning, cancer-incremental learning, and institute-incremental learning scenarios.

### Inatsll MCSA
To install from the development branch run
```
git clone git@github.com:Earth1998/MCSA.git
cd MCSA/
```

Finally, configure the defalut path of the above tool and the database in `conf.py`. You can change the path of the tool and database by configuring `conf.py` as needed.

## Usage
To use MCSA, run
```
python main.py --config=exps/settings.json
```

## Feedback
If you have questions on how to use MCSA, feel free to raise questions in the [discussions section](https://github.com/Earth1998/MCSA/discussions). If you identify any potential bugs, feel free to raise them in the [issuetracker](https://github.com/Earth1998/MCSA/issues).

In addition, if you have any further questions about MCSA, please feel free to contact us [thquan@bliulab.net]

## Citation

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
## Acknowledgement
