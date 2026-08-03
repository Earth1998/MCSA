import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from mwe import MWE


class GDSCDataset(Dataset):
    def __init__(self, version='gdsc2', platform='rma', geneset_file='data/hgnc_csv/depmap_gdsc_tcga_gene.csv'):
        drug_df = pd.read_csv("data/gdsc_csv/gdsc_smiles.csv")
        drug_df = drug_df.dropna(subset=["smiles"])
        
        if version == "gdsc1":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC1_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC1"])]
        elif version == "gdsc2":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC2_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC2"])]
        
        drugid_smi_dict = dict(zip(drug_df["Drug Id"], drug_df["smiles"]))
        gdsc_df["smiles"] = gdsc_df["DRUG_ID"].map(drugid_smi_dict)
        gdsc_df = gdsc_df.dropna(subset=["smiles"])
        
        if platform == "rma":
            gex_df = pd.read_csv("data/gdsc_csv/cell_line_gex_rma.csv", index_col=0)
            cosmic_ids = [row.split(".")[1] for row in gex_df.index.to_list()]
            # gdsc_df["COSMIC_ID"] = gdsc_df["COSMIC_ID"].astype('str')
            gdsc_df = gdsc_df[gdsc_df["COSMIC_ID"].astype('str').isin(cosmic_ids)]
            self.smiles = gdsc_df["smiles"].to_list()
            self.cells = ["DATA." + str(idx) for idx in gdsc_df["COSMIC_ID"].to_list()]
            self.labels = gdsc_df["LN_IC50"].to_list()
        elif platform == "tpm":
            gex_df = pd.read_csv("data/depmap_csv/depmap_gex_tpm.csv", index_col=0)
            depmap_info = pd.read_csv("data/depmap_csv/Model.csv", index_col=0)
            depmap_sanger_dict = depmap_info.dropna(subset=["SangerModelID"])["SangerModelID"].to_dict()
            gex_df.index = gex_df.index.map(depmap_sanger_dict)
            gex_df = gex_df[~gex_df.index.isna()]
            gdsc_df = gdsc_df[gdsc_df["SANGER_MODEL_ID"].isin(gex_df.index)]
            self.smiles = gdsc_df["smiles"].to_list()
            self.cells = gdsc_df["SANGER_MODEL_ID"].to_list()
            self.labels = gdsc_df["LN_IC50"].to_list()
        
        geneset_list = pd.read_csv(geneset_file)["gene"].to_list()
        gex_df = gex_df[geneset_list]
        gex_df = gex_df.rank(axis=1, pct=True)
        
        self.tokenizer = MWE("data/vocab_csv/subword.csv")
        self.gex = gex_df
        
        # gdsc_df.to_csv(f"data/gdsc_csv/{version}_{platform}.csv", index=False)
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, index):
        smi = self.smiles[index]
        tok = np.array(self.tokenizer.smiles_to_token(smi))
        cell = self.cells[index]
        rna = self.gex.loc[cell].to_numpy()
        label = np.array([self.labels[index]])
        return {
            "tok": tok,
            "rna": rna,
            "label": label
        }


def _pad_fn(a, value):
    a = [np.array(_) for _ in a]
    max_shape = np.max([_.shape for _ in a], axis=0)
    na = []
    for x in a:
        pad_shape = [(0, l2 - l1) for l1, l2 in zip(x.shape, max_shape)]
        na.append(np.pad(x, pad_shape, mode="constant", constant_values=value))
    return np.stack(na)


pad_values = {
    "tok": 3000,
    "rna": 0.0,
    "label": -1
}


def collate_fn(batch):
    ret = {}
    for key in batch[0].keys():
        ret[key] = _pad_fn([_[key] for _ in batch], pad_values[key])
    for key in ret.keys():
        if key == "tok":
            ret[key] = torch.tensor(ret[key], dtype=torch.long)
        if key == "rna":
            ret[key] = torch.tensor(ret[key], dtype=torch.float)
        if key == "label":
            ret[key] = torch.tensor(ret[key], dtype=torch.float)
    return ret
