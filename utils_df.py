import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from mwe import MWE


class TCGADataset(Dataset):
    def __init__(self, geneset_file):
        drug_df = pd.read_csv("data/tcga_gex/tcga_records.csv")
        gex_df = pd.read_csv("data/tcga_gex/tcga_gex_tpm.csv", index_col=0)
        gex_index_list = [idx[:12] for idx in gex_df.index.to_list()]
        gex_df.index = gex_index_list
        gex_df = gex_df[~gex_df.index.duplicated(keep='first')]
        
        geneset_list = pd.read_csv(geneset_file)["gene"].to_list()
        gex_df = gex_df[geneset_list]
        gex_df = gex_df.rank(axis=1, pct=True)
        
        drug_df = drug_df[drug_df["bcr_patient_barcode"].isin(gex_df.index)]
        drug_df["label"] = drug_df["measure_of_response"].map({"Complete Response": 1, "Partial Response": 1, "Clinical Progressive Disease": 0, "Stable Disease": 0})
        smiles_counts = drug_df.groupby('smiles')['smiles'].transform('count')
        drug_df = drug_df[smiles_counts >= 20]
        
        self.smiles = drug_df["smiles"].to_list()
        self.cells = drug_df["bcr_patient_barcode"].to_list()
        self.labels = drug_df["label"].to_list()
        
        self.tokenizer = MWE("data/vocab_csv/subword.csv")
        self.gex = gex_df
    
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
