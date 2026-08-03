from abc import abstractmethod
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pmoe import PMOE
from prototype import Prototyper
from pipgim import PGIM, PIPM


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, bias=None, mask_attn=None):
        attn = torch.matmul(q / self.scale, k.transpose(-1, -2))
        
        if bias is not None:
            attn += bias

        if mask_attn is not None:
            attn += mask_attn
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_key, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        if d_model % n_head != 0:
            raise ValueError(
                "The hidden size is not a multiple of the number of attention heads"
            )
        self.n_head = n_head
        self.d_k = d_key // n_head
        self.fc_query = nn.Linear(d_model, d_key, bias=False)
        self.fc_key = nn.Linear(d_model, d_key, bias=False)
        self.fc_value = nn.Linear(d_model, d_key, bias=False)

        self.attention = ScaledDotProductAttention(
            scale=self.d_k**0.5, dropout=dropout
        )
        self.fc_out = nn.Linear(d_key, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (self.n_head, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, x, bias=None, mask_attn=None):
        q = self.transpose_for_scores(self.fc_query(x))
        k = self.transpose_for_scores(self.fc_key(x))
        v = self.transpose_for_scores(self.fc_value(x))

        x, attn_weight = self.attention(q, k, v, bias=bias, mask_attn=mask_attn)
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        x = self.dropout(self.fc_out(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_key, n_head, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model, d_key=d_key, n_head=n_head, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.pmoe = PMOE(d_model=d_model, router=None, num_adapter=10)

    def forward(self, x, bias, mask_attn, t):
        x = x + self.attn(self.norm1(x), bias, mask_attn = mask_attn)
        x = x + self.ffn(self.norm2(x))
        x = x + self.pmoe(x, t)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(**kwargs) for _ in range(n_layer)]
        )

    def forward(self, x, mask, mask_attn_embed, t):
        bias = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device=x.device)
        bias[mask.unsqueeze(1).expand_as(bias)] = -10000
        bias = bias.unsqueeze(1)
        # mask_attn_embed = mask_attn_embed.to(dtype=x.dtype)

        embs = []
        
        for module in self.layers:
            x = module(x, bias, mask_attn_embed, t)
            embs.append(x[:, 0])
        return x, embs


class Transformer(BaseModel):
    def __init__(self):
        super(Transformer, self).__init__()
        self.d_model = 256
        self.n_cls = 1
        self.word_embeddings = nn.Sequential(
            nn.Embedding(3002, self.d_model, padding_idx=3000), nn.Dropout(0.1)
        )
        self.position_embeddings = nn.Sequential(
            nn.Embedding(3002, self.d_model), nn.Dropout(0.1)
        )
        
        self.encoder = TransformerEncoder(
            n_layer=6,
            d_model=self.d_model,
            d_key=self.d_model,
            n_head=self.d_model // 16,
            dim_feedforward=self.d_model * 4,
            dropout=0.1
        )
        
        self.fc = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )

    def forward(self, input_ids, t=0):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings =  self.word_embeddings(input_ids)
        position_embeddings =  self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        
        cls_id = torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        cls_id[:] = 3001
        cls_emb = self.word_embeddings(cls_id) + self.position_embeddings(cls_id)
        
        embeddings = torch.cat([cls_emb, embeddings], dim=1)
        
        input_ids = torch.cat([cls_id, input_ids], dim=1)
        mask = input_ids == 3000

        x, emb = self.encoder(embeddings, mask, mask_attn_embed=None, t=t)
        x = self.fc(x[:, 0])

        return x


class DRModel(BaseModel):
    def __init__(self):
        super(DRModel, self).__init__()
        dropout = 0.1
        self.drug_encoder = Transformer()
        self.cell_encoder = nn.Sequential(
            nn.Linear(15743, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.predictor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(256, 1)
        self.pmoe = PMOE(d_model=512, router=None, num_adapter=10)
        self.router = None
        self.pgim = PGIM(input_dim=15743, hidden_dim=32, attn_dim=2)
        # self.pipm = PIPM(input_dim=15743, hidden_dim=32, attn_dim=2, num_pm=10)
    
    def set_router(self, model):
        self.router = Prototyper(model=model)
    
    def get_emb(self, drug, cell, t=0):
        drug_repr = self.drug_encoder(drug, t)
        cell_repr = self.cell_encoder(cell)
        v = torch.cat([drug_repr, cell_repr], dim=1)
        v = v + self.pmoe(v, t)
        return v
    
    def get_attn(self, drug, cell, t=0):
        cell, attn_score = self.pgim(cell)
        # if self.router is not None and t is None:
        #     t = self.router.calculate(drug, cell)
        # cell, attn_score = self.pipm(cell, t)
        return attn_score
    
    def forward(self, drug, cell, t):
        cell, _ = self.pgim(cell)
        if self.router is not None and t is None:
            t = self.router.calculate(drug, cell)
        # cell, _ = self.pipm(cell, t)
        drug_repr = self.drug_encoder(drug, t)
        cell_repr = self.cell_encoder(cell)
        v = torch.cat([drug_repr, cell_repr], dim=1)
        v_p = self.pmoe(v, t)
        v = v + v_p
        v = self.predictor(v)
        pred = self.out(v)
        return pred


class AutoEncoder(BaseModel):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        dropout = 0.1
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
