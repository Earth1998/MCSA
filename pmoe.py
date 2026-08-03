import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, d_model):
        super(Adapter, self).__init__()
        self.d_model = d_model
        
        self.in_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.in_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x):
        residual = x
        
        x = self.in_proj(x)
        x = self.act_func(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PMOE(nn.Module):
    def __init__(self, d_model, router, num_adapter):
        super(PMOE, self).__init__()
        self.router = router
        self.adapter_list = nn.ModuleList()
        for i in range(num_adapter):
            self.adapter_list.append(Adapter(d_model=d_model))
    
    def forward(self, x, t):
        x = self.adapter_list[t](x)
        return x
