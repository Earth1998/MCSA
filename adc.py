import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from attack import Attack

class CustomDataset(Dataset):
    def __init__(self, rna, drug, target):
        self.rna = rna
        self.drug = drug
        self.target = target
    
    def __len__(self):
        return len(self.rna)
    
    def __getitem__(self, index):
        gene_expression = self.rna[index]
        drug_token = self.drug[index]
        IC50_value = self.target[index]
        return gene_expression, drug_token, IC50_value


# generate adversarial drugs and rnaseq profiles
def adversarial_sample_generation(train_loader, epoch, batch_size, alpha, model, sample_limit, task_id):
    for k, (rna, drug, target) in enumerate(train_loader):
        if k == 0:
            rna_min = rna.min()
            rna_max = rna.max()
            drug_min = drug.min()
            drug_max = drug.max()
        else:
            if rna.min() < rna_min:
                rna_min = rna.min()
            if rna.max() > rna_max:
                rna_max = rna.max()
            if drug.min() < drug_min:
                drug_min = drug.min()
            if drug.max() > drug_max:
                drug_max = drug.max()
    
    rna_, drug_, target_, feats = [], [], [], []
    for rna, drug, target in train_loader:
        # rna, drug, target = rna.cpu(), drug.cpu(), target.cpu()
        print(model.training)
        with torch.no_grad():
            rna_.append(rna)
            drug_.append(drug)
            target_.append(target)
            feats.append(model.encode(rna, drug))
    
    rna_ = torch.cat(rna_, dim=0)
    drug_ = torch.cat(drug_, dim=0)
    target_ = torch.cat(target_, dim=0)
    feats = torch.cat(feats, dim=0)

    rr, dd, tt = [], [], []
    for idx in range(0, task_id):
        for pro_idx in range(model._protos[idx].shape[0]):
            d = torch.cdist(feats, model._protos[idx][pro_idx].unsqueeze(0)).squeeze()
            closest = torch.argsort(d)[:sample_limit].cpu()
            rna_top = rna_[[closest]]
            drug_top = drug_[[closest]]
            target_top = target_[[closest]]

            pro_idx_dataset = CustomDataset(rna_top, drug_top, target_top)
            loader = DataLoader(pro_idx_dataset, batch_size=int(sample_limit), shuffle=False)
            
            attack = Attack(model, None, alpha, loader, model._protos[idx][pro_idx], model.device, epoch, rna_min, rna_max, drug_min, drug_max, idx)
            r_, d_, t_ = attack.run()
            rr.append(r_)
            dd.append(d_)
            tt.append(t_)
    
    rr = torch.cat(rr, dim=0)
    dd = torch.cat(dd, dim=0)
    tt = torch.cat(tt, dim=0)

    idx_dataset = TensorDataset(rr, dd, tt)
    idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False)
    return idx_loader
