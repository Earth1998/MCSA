import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from pmodel import DRModel
from utils import GDSCDataset, collate_fn
from splitter import pathway_splitter
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(model, val_loader, device, t=None):
    
    model.eval()
    
    epoch_loss = 0.0
    total_samples = 0
    label_list = []
    output_list = []
    
    loss_fn = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in val_loader:
            tok = batch["tok"].to(device)
            rna = batch["rna"].to(device)
            label = batch["label"].to(device)
            
            num_samples = rna.shape[0]
            total_samples += num_samples
            
            pred = model(tok, rna, t=t) 
            loss = loss_fn(pred, label)
            
            epoch_loss += loss.item() * num_samples
            
            label_list.append(label.cpu().detach().numpy().ravel())
            output_list.append(pred.cpu().detach().numpy().ravel())
    
    y_true = np.concatenate(label_list)
    y_output = np.concatenate(output_list)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_output))
    r2 = r2_score(y_true, y_output)
    pcc, _ = pearsonr(y_true, y_output)
    
    return {
        "loss": epoch_loss / total_samples,
        "rmse": rmse,
        "r2": r2,
        "pcc": pcc
    }


def evaluate_saved_model(seed, model_path):
    set_seed(seed)
    
    gdscdataset = GDSCDataset()
    pathway_index_list = pathway_splitter(shuffle=True)
    
    loader_list = []
    for pathway_dict in pathway_index_list:
        test_idx = pathway_dict["test"]
        test_dataset = Subset(gdscdataset, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn, drop_last=True)
        loader_list.append(test_loader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    drmodel = checkpoint["drmodel"]
    drmodel.to(device)
    drmodel.eval()
    
    rmse_list = []
    pcc_list = []
    
    for test_id, test_loader in enumerate(loader_list):
        
        metrics = validate(drmodel, test_loader, device, t=None)
        
        rmse_list.append(metrics["rmse"])
        pcc_list.append(metrics["pcc"])

    avg_rmse = np.mean(rmse_list)
    avg_pcc = np.mean(pcc_list)
    
    print("-" * 50)
    print(f"RMSE: {avg_rmse:.4f}")
    print(f"PCC : {avg_pcc:.4f}")
    
    return avg_rmse, avg_pcc

if __name__ == "__main__":
    
    seed_list = [42, 43, 44, 45, 46]
    all_rmse = []
    all_pcc = []
    
    for seed in seed_list:
        model_path = f"model_save/seed_{seed}/final_model.pt"
        try:
            rmse, pcc = evaluate_saved_model(seed, model_path)
            all_rmse.append(rmse)
            all_pcc.append(pcc)
        except Exception as e:
            print(f"Failed for seed {seed}: {e}")
            
    print("\n" + "="*50)
    print(f"RMSE : {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
    print(f"PCC  : {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
