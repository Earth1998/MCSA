import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from mwe import MWE
from pmodel import DRModel, AutoEncoder 


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
        
        if key == "smi":
            ret[key] = [_[key] for _ in batch]
            continue
        ret[key] = _pad_fn([_[key] for _ in batch], pad_values[key])
        
    for key in ret.keys():
        if key == "tok":
            ret[key] = torch.tensor(ret[key], dtype=torch.long)
        elif key == "rna":
            ret[key] = torch.tensor(ret[key], dtype=torch.float)
        elif key == "label":
            ret[key] = torch.tensor(ret[key], dtype=torch.float)
    return ret


class TCGAEvalDataset(Dataset):
    def __init__(self, df_path, geneset_file):
        self.df = pd.read_csv(df_path)
        self.smiles = self.df["smiles"].to_list()
        self.cells = self.df["cell"].to_list()
        self.labels = self.df["label"].to_list()
        
        gex_df = pd.read_csv("data/tcga_gex/tcga_gex_tpm.csv", index_col=0)
        gex_index_list = [idx[:12] for idx in gex_df.index.to_list()]
        gex_df.index = gex_index_list
        gex_df = gex_df[~gex_df.index.duplicated(keep='first')]
        
        geneset_list = pd.read_csv(geneset_file)["gene"].to_list()
        gex_df = gex_df[geneset_list]
        gex_df = gex_df.rank(axis=1, pct=True)
        self.gex = gex_df
        
        self.tokenizer = MWE("data/vocab_csv/subword.csv")
    
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
            "label": label,
            "smi": smi
        }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    print("Loading dataset from DataFrame...")
    test_csv_path = "data/saved_splits/tcga_test_set.csv"
    geneset_file = "data/hgnc_csv/depmap_gdsc_tcga_gene.csv"
    
    eval_dataset = TCGAEvalDataset(df_path=test_csv_path, geneset_file=geneset_file)
    print(f"Dataset loaded. Total test samples: {len(eval_dataset)}")
    
    print("Loading saved model...")
    model_path = os.path.join("model_z", "model.pt")
    checkpoint = torch.load(model_path, map_location=device)
    drmodel = checkpoint["drmodel"]
    drmodel = drmodel.to(device)
    drmodel.eval()
    
    drug_data = {}
    
    print("Running inference...")
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            single_sample = eval_dataset[i]
            smi = single_sample["smi"]
            
            batch = collate_fn([single_sample])
            
            tok = batch["tok"].to(device)
            rna = batch["rna"].to(device)
            true_label = batch["label"].to(device)
            
            pred = drmodel(tok, rna, t=0)
            prob = torch.sigmoid(pred).item()
            label_val = true_label.item()
            
            if smi not in drug_data:
                drug_data[smi] = {"y_true": [], "y_prob": []}
            
            drug_data[smi]["y_true"].append(label_val)
            drug_data[smi]["y_prob"].append(prob)
            
    print("\n========== Per-Drug Evaluation ==========")
    results = {}
    for smi, data in drug_data.items():
        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])
        
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            results[smi] = {"AUROC": auroc, "AUPRC": auprc, "Count": len(y_true)}
            print(f"Drug SMILES: {smi[:30]:<30} | Count: {len(y_true):<4} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
        else:
            print(f"Drug SMILES: {smi[:30]:<30} | Count: {len(y_true):<4} | Skipped (Only one class present)")

if __name__ == "__main__":
    main()
