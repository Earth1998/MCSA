import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from mwe import MWE
from pmodel import DRModel, AutoEncoder
import pickle
from collections import defaultdict


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

        # with open("data/tcga_gex/tcga_gex_tpm.pkl", 'rb') as f:
        #     gex_df = pickle.load(f)
        
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


def evaluate_single_seed(seed, device, batch_size=64):
    
    # print(f"\n[{seed}] Loading dataset and model for seed {seed}...")
    test_csv_path = f"data/saved_splits/tcga_test_set_{seed}.csv"
    geneset_file = "data/hgnc_csv/depmap_gdsc_tcga_gene.csv"

    eval_dataset = TCGAEvalDataset(df_path=test_csv_path, geneset_file=geneset_file)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model_path = os.path.join("model_z", f"model_{seed}.pt")
    checkpoint = torch.load(model_path, map_location=device)
    drmodel = checkpoint["drmodel"]
    drmodel = drmodel.to(device)
    drmodel.eval()

    drug_data = defaultdict(lambda: {"y_true": [], "y_prob": []})
    all_y_true = []
    all_y_prob = []

    # print(f"[{seed}] Running inference...")
    with torch.no_grad():
        for batch in dataloader:
            tok = batch["tok"].to(device)
            rna = batch["rna"].to(device)
            true_label = batch["label"].to(device)
            smis = batch["smi"]

            pred = drmodel(tok, rna, t=0)
            prob = torch.sigmoid(pred).cpu().view(-1).numpy().tolist()
            label_val = true_label.cpu().view(-1).numpy().tolist()

            all_y_true.extend(label_val)
            all_y_prob.extend(prob)

            for i, smi in enumerate(smis):
                drug_data[smi]["y_true"].append(label_val[i])
                drug_data[smi]["y_prob"].append(prob[i])

    # 1. 计算当前 seed 下每个药物的指标
    seed_drug_metrics = {}
    for smi, data in drug_data.items():
        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])

        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            seed_drug_metrics[smi] = {"AUROC": auroc, "AUPRC": auprc, "Count": len(y_true)}

    # 2. 计算当前 seed 下总体的指标
    overall_auroc, overall_auprc = None, None
    if len(np.unique(all_y_true)) > 1:
        overall_auroc = roc_auc_score(all_y_true, all_y_prob)
        overall_auprc = average_precision_score(all_y_true, all_y_prob)
        print(f"[{seed}] Overall AUROC: {overall_auroc:.4f} | Overall AUPRC: {overall_auprc:.4f}")
    else:
        print(f"[{seed}] Skipped overall evaluation (Only one class present).")

    return overall_auroc, overall_auprc, seed_drug_metrics


def main():
    
    seeds = [42, 43, 44, 45, 46] 

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Record the results for all seeds.
    all_seeds_overall_auroc = []
    all_seeds_overall_auprc = []
    
    # Record the AUROC and AUPRC for each drug across the various seeds.
    # Data Structures: {smi: {"AUROC": [v1, v2, ...], "AUPRC": [v1, v2, ...], "Count": [c1, c2, ...]}}
    all_seeds_drug_metrics = defaultdict(lambda: {"AUROC": [], "AUPRC": [], "Count": []})

    # Iteratively evaluate each seed.
    for seed in seeds:
        try:
            auroc, auprc, drug_metrics = evaluate_single_seed(seed, device, batch_size=1)
            
            if auroc is not None and auprc is not None:
                all_seeds_overall_auroc.append(auroc)
                all_seeds_overall_auprc.append(auprc)
                
            for smi, metrics in drug_metrics.items():
                all_seeds_drug_metrics[smi]["AUROC"].append(metrics["AUROC"])
                all_seeds_drug_metrics[smi]["AUPRC"].append(metrics["AUPRC"])
                all_seeds_drug_metrics[smi]["Count"].append(metrics["Count"])
                
        except Exception as e:
            print(f"Error evaluating seed {seed}: {e}")

    # ========================== FINAL AGGREGATED RESULTS ==========================
    print("\n\n============================================================")
    print("                     FINAL AGGREGATED RESULTS                 ")
    print("============================================================")

    # Overall Mean ± Std
    if all_seeds_overall_auroc:
        mean_auroc = np.mean(all_seeds_overall_auroc)
        std_auroc = np.std(all_seeds_overall_auroc)
        mean_auprc = np.mean(all_seeds_overall_auprc)
        std_auprc = np.std(all_seeds_overall_auprc)
        
        print("\n[OVERALL PERFORMANCE]")
        # print(f"Seeds tested : {len(all_seeds_overall_auroc)} ({seeds})")
        print(f"Overall AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
        print(f"Overall AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")
    else:
        print("\n[OVERALL PERFORMANCE] No valid overall results to aggregate.")

    # Per-drug Mean ± Std
    print("\n[PER-DRUG PERFORMANCE]")
    print(f"{'Drug SMILES':<30} | {'Seed Hits':<9} | {'Mean Count':<10} | {'AUROC (Mean ± Std)':<20} | {'AUPRC (Mean ± Std)':<20}")
    print("-" * 110)
    
    # Sort the results in descending order based on the average AUROC
    sorted_drugs = sorted(
        all_seeds_drug_metrics.items(), 
        key=lambda x: np.mean(x[1]["AUROC"]) if x[1]["AUROC"] else 0, 
        reverse=True
    )

    for smi, metrics in sorted_drugs:
        if not metrics["AUROC"]:
            continue
            
        mean_count = np.mean(metrics["Count"])
        mean_d_auroc = np.mean(metrics["AUROC"])
        std_d_auroc = np.std(metrics["AUROC"])
        mean_d_auprc = np.mean(metrics["AUPRC"])
        std_d_auprc = np.std(metrics["AUPRC"])
        seed_hits = len(metrics["AUROC"]) # The drug successfully calculated the metrics across several seeds.
        
        smi_display = smi[:27] + "..." if len(smi) > 30 else smi
        
        print(f"{smi_display:<30} | {seed_hits:<9} | {mean_count:<10.1f} | "
              f"{mean_d_auroc:.4f} ± {std_d_auroc:.4f} | {mean_d_auprc:.4f} ± {std_d_auprc:.4f}")


if __name__ == "__main__":
    main()
