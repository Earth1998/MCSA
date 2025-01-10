import os
import sys
import numpy as np
import random
from copy import deepcopy
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch._six import inf
import pandas as pd
from tqdm.contrib import tenumerate


def fisher_matrix_diag(dataloader, model):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    # for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
    for batch_id, (rna, drug, target) in enumerate(tqdm(dataloader)):
        
        # Forward and backward
        model.zero_grad()
        result = model(rna, drug)
        outputs = result[0]
        loss = criterion(outputs, target)
        loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=rna.shape[0]*p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n,_ in model.named_parameters():
            fisher[n]=fisher[n]/len(dataloader.dataset)
    return fisher


def set_fisher(model, dataloader, t):
    if t>0:
        fisher_old={}
        for n,_ in model.named_parameters():
            fisher_old[n]=model.fisher[n].clone()
    model.fisher=fisher_matrix_diag(dataloader,model)
    if t>0:
        # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
        for n,_ in model.named_parameters():
            model.fisher[n] = (model.fisher[n] + fisher_old[n] * t) / (t + 1)


def compute_emb_mean(model, dataloader, t):
    vectors = []
    model.eval()
    with torch.no_grad():
        for batch_id, (rna, drug, target) in enumerate(tqdm(dataloader)):
            result=model(rna,drug)
            z = result[1]
            z = z.cpu().numpy()
            vectors.append(z)
        vectors = np.concatenate(vectors)
        emb_mean = np.mean(vectors, axis=0)
        emb_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(emb_mean.shape[-1])*1e-5
        model.emb_means_.append(emb_mean)
        model.emb_covs_.append(emb_cov)


def attention_reg(A_old, A):
    mask = (A_old > 0.35).float()
    diff = F.mse_loss(A * mask, A_old * mask)
    return diff


# import torch
# import torch.nn as nn

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # register weight
#         self.weight = nn.Parameter(torch.randn(10, 10))
#         self.register_parameter('weight', self.weight)
#         self.old_param = {}
#         for n, p in self.named_parameters():
#             self.old_param[n] = p.data.clone().detach()

# model = MyModel()
# for name, param in model.named_parameters():
#     print(name, param.shape)

# print('\n', '*'*40, '\n')

# for key, val in model.state_dict().items():
#     print(key, val.shape)

# x = torch.tensor(2., requires_grad = True)
# print("x:", x)

# # define a function y
# with torch.no_grad():
#     for i in range(1):
#         x = x / 2
# print("x:", x)

# # check gradient for Y
# print("x.requires_grad:", x.requires_grad)


def calculate_importance(model, dataloader):
    out = {}
    # initialize Omega(Ω) matrix（with 0）by adding previous guards
    for n, p in model.named_parameters():
        out[n] = p.clone().detach().fill_(0)
        for prev_guard in model.previous_guards_list:
            if prev_guard:
                out[n] += prev_guard[n]
    
    model.eval()
    if dataloader is not None:
        number_data = len(dataloader)
        for batch_id, (rna, drug, target) in enumerate(tqdm(dataloader)):
            model.zero_grad()
            pred = model(rna, drug)[0]
            ##### generate Omega(Ω) matrix.  #####   
            # network output L2 norm square grads
            loss = torch.mean(torch.sum(pred ** 2, axis=1))
            loss.backward()
            for n, p in model.named_parameters():
                out[n].data += torch.sqrt(p.grad.data ** 2) / number_data
    
    out = {n: p for n, p in out.items()}
    return out

def fisher_matrix_diag_fusion(model_a, model_b, model, dataloader):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model_a.eval()
    model_b.eval()
    model.train()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    # for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
    for batch_id, (rna, drug, target) in enumerate(tqdm(dataloader)):
        
        # Forward and backward
        model.zero_grad()
        with torch.no_grad():
            z_a = model_a.feature_extraction(rna,drug)
            z_b = model_b.feature_extraction(rna,drug)
        z_ab = torch.cat((z_a, z_b), dim=1)
        result=model(z_ab)
        outputs = result[0]
        loss = criterion(outputs, target)
        loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=rna.shape[0]*p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n,_ in model.named_parameters():
            fisher[n]=fisher[n]/len(dataloader.dataset)
    return fisher


def set_fisher_fusion(model_a, model_b, model, dataloader, t):
    if t>0:
        fisher_old={}
        for n,_ in model.named_parameters():
            fisher_old[n]=model.fisher[n].clone()
    model.fisher=fisher_matrix_diag_fusion(model_a, model_b, model, dataloader)
    if t>0:
        # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
        for n,_ in model.named_parameters():
            model.fisher[n] = (model.fisher[n] + fisher_old[n] * t) / (t + 1)
