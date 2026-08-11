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

The main dependencies used in this project are as follows:

```
scikit-learn 1.8.0
scipy 1.17.1
numpy 2.2.6
pandas 2.3.3
openpyxl 3.1.5
nltk 3.9.4
torch 2.5.1
```

For more information on torch versions, see the [pytorch installation documentation](https://pytorch.org/).

### Datasets

The datasets can be download from [here](https://doi.org/10.5281/zenodo.21715575).
Data directory structure:

```text
cd data/gdsc_csv/
unzip gdsc.zip

data/
├── depmap_csv/
│   └── depmap_gex_tpm.csv
├── gdsc_csv/
│   └── cell_line_gex_rma.csv
│   └── gdsc_smiles.csv
│   └── GDSC2_fitted_dose_response_27Oct23.xlsx
│   └── gdsc2_rma.csv
├── tcga_gex/
│   └── tcga_gex_tpm.csv
└── ...
```

### Modules

```
The implementations of the continual self-supervised adversarial learning/perturbation module (CSSAL) can be found in 'uad.py' and 'utils_adv.py'.
The implementations of the plug-in pharmacogenomic interpretable module (PIPGIM) can be found in 'pipgim.py'.
The implementations of the interpretability-consistency regularization (CREG/IRST) can be found in 'pipgim.py'.
The implementations of the prototype generation can be found in 'prototype.py'.
The implementations of the prototype-based mixture-of-expert module (PMOE) can be found in 'pmoe.py'.
The implementations of the data loading can be found in 'utils.py', 'utils_bf.py', 'utils_df.py' and 'splitter.py'.
The implementations of the drug encoder and autoencoder can be found in 'pmodel.py'.
```

### Usage

Please organize the datasets according to the data directory structure. The models can be download from [here](https://drive.google.com/file/d/11Ms3TFG4YiOgjI5GBeoTzTzjiyYXQdeK/view?usp=drive_link).

```
tar -zxf model.tar.gz
python run.py
```

Please organize the datasets according to the data directory structure. For tcga, the models can be download from [here](https://drive.google.com/file/d/1rAZWQood2527KeeuqhLpkcxhi0g07Psw/view?usp=sharing).

```
tar -zxf model_z.tar.gz
python run_t.py
```

### Contact

If you have any questions, please feel free to contact thquan@bliulab.net.


## Acknowledgement

We are grateful to the anonymous reviewers for their valuable comments and suggestions.
