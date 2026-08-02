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
scipy 1.17.1
numpy 2.2.6
pandas 2.3.3
openpyxl 3.1.5
nltk 3.9.4
torch 2.5.1
```

For more information on torch versions, see the [pytorch installation documentation](https://pytorch.org/).

### Datasets

The data used for drug response prediction can be download from [here](https://doi.org/10.5281/zenodo.21715575).

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
