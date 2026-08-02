# MCSA: An interpretable multi-context self-alignment framework for continual drug response prediction and pharmacogenomic analysis

This is the official code repository of MCSA.

## Installation

### Create conda environment

```
conda create -n mcsa python=3.12
conda activate mcsa
```
For installing conda, please refer to https://www.anaconda.com/download/success.

### Requirements
The main dependencies used in this project are as follows (for more information, please see the `environment.yaml` file):

```
scikit-learn 1.8.0
scipy 1.9.3
numpy 1.24.3
pandas 2.3.3
openxyl 3.1.5
nltk 3.9.4
torch 2.5.1
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
