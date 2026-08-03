import torch
import torch.nn as nn
import torch.nn.functional as F


class PGIM(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_dim):
        super(PGIM, self).__init__()
        
        self.extractor = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.Parameter(torch.randn(input_dim, hidden_dim, attn_dim) * 0.1)
        self.bias = nn.Parameter(torch.ones(input_dim, attn_dim) * 0.1)
        self.scale = 0.1
    
    def forward(self, x):
        E = torch.tanh(self.extractor(x))
        attn_logits = torch.einsum('be, iea -> bia', E, self.attn) + self.bias
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_score = attn_probs[:, :, 1]
        x = x + x * attn_score * self.scale
        # x = x * attn_score
        return x, attn_score


class PIPM(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_dim, num_pm):
        super(PIPM, self).__init__()
        
        self.pgim_list = nn.ModuleList()
        for i in range(num_pm):
            self.pgim_list.append(PGIM(input_dim, hidden_dim, attn_dim))
    
    def forward(self, x, t):
        x, attn_score = self.pgim_list[t](x)
        return x, attn_score


def reg_irst(old_a, a, t):
    mask = old_a > t
    diff = F.mse_loss(a[mask], old_a[mask])
    return diff
