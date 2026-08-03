import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class Prototyper():
    def __init__(self, model=None):
        self.model = model
        self.protos = {}
    
    def build_protos(self, t, dataloader, num_protos, device):
        self.model.eval()
        feats = []
        
        for idx, batch in enumerate(dataloader):
            with torch.no_grad():
                
                tok = batch["tok"].to(device)
                rna = batch["rna"].to(device)
                label = batch["label"].to(device)
                
                t_repr = self.model.get_emb(tok, rna)
                feats.append(t_repr.detach().cpu().numpy())
        
        feats = np.concatenate(feats, axis=0)
        
        km = KMeans(n_clusters=num_protos, n_init='auto').fit(feats)
        centers = km.cluster_centers_
        
        self.protos[t] = centers
    
    def set_model(self, model):
        self.model = model
    
    def calculate(self, tok, rna):
        t_repr = self.model.get_emb(tok, rna)
        
        best_t = None
        min_mean_diff = float('inf')
        
        for t, protos in self.protos.items():
            protos_tensor = torch.tensor(protos, dtype=t_repr.dtype, device=t_repr.device)
            dists = torch.cdist(t_repr, protos_tensor)
            mean_diff = dists.min(dim=1)[0].mean().item()
            if mean_diff < min_mean_diff:
                min_mean_diff = mean_diff
                best_t = t
        return best_t
