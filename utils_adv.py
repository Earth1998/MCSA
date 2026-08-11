import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialGenerator:
    """
    针对输入 (tok_emb, rna) 在 Embedding 空间上生成 FGSM/PGD 风格的对抗扰动
    """
    def __init__(self, model, alpha=0.01, eps=0.05, steps=3):
        self.model = model
        self.alpha = alpha  # 每步扰动步长
        self.eps = eps      # 最大扰动范围 (L_inf norm bound)
        self.steps = steps  # PGD 迭代步数

    def generate(self, tok, rna, label, device):
        self.model.eval()
        
        # 提取离散 token 的连续 Embedding 特征 (B, L, D)
        with torch.no_grad():
            seq_len = tok.size(1)
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand_as(tok)
            orig_emb = self.model.drug_encoder.word_embeddings(tok) + \
                       self.model.drug_encoder.position_embeddings(pos_ids)
            
            # 引入 CLS Token 的 Embedding 拼接（完全对齐 DRModel 内部实现）
            cls_id = torch.ones((tok.shape[0], 1), dtype=tok.dtype, device=device) * 3001
            cls_emb = self.model.drug_encoder.word_embeddings(cls_id) + \
                      self.model.drug_encoder.position_embeddings(cls_id)
            full_emb = torch.cat([cls_emb, orig_emb], dim=1)
            
            # 构建 mask
            full_tok = torch.cat([cls_id, tok], dim=1)
            mask = (full_tok == 3000)

        # 针对嵌入空间和 RNA 施加对抗扰动
        delta_emb = torch.zeros_like(full_emb, requires_grad=True, device=device)
        delta_rna = torch.zeros_like(rna, requires_grad=True, device=device)

        loss_fn = nn.MSELoss()

        for _ in range(self.steps):
            adv_emb = full_emb + delta_emb
            adv_rna = rna + delta_rna
            
            # 穿透 Transformer Encoder & Predictor 得到输出
            # 利用 DRModel 现有的子组件拼接前向逻辑
            x, _ = self.model.drug_encoder.encoder(adv_emb, mask, mask_attn_embed=None, t=0)
            drug_repr = self.model.drug_encoder.fc(x[:, 0])
            
            cell_repr, _ = self.model.pgim(adv_rna)
            cell_repr = self.model.cell_encoder(cell_repr)
            
            v = torch.cat([drug_repr, cell_repr], dim=1)
            v = v + self.model.pmoe(v, t=0)
            v = self.model.predictor(v)
            pred = self.model.out(v)

            loss = loss_fn(pred, label)
            loss.backward()

            # 梯度方向最大化 Loss (对抗攻击)
            grad_emb = delta_emb.grad.detach()
            grad_rna = delta_rna.grad.detach()

            delta_emb = delta_emb.detach() + self.alpha * grad_emb.sign()
            delta_emb = torch.clamp(delta_emb, -self.eps, self.eps).requires_grad_(True)

            delta_rna = delta_rna.detach() + self.alpha * grad_rna.sign()
            delta_rna = torch.clamp(delta_rna, -self.eps, self.eps).requires_grad_(True)

            self.model.zero_grad()

        return full_emb + delta_emb.detach(), rna + delta_rna.detach(), mask


class Generator(nn.Module):
    """根据药物/RNA 隐向量生成增广特征"""
    def __init__(self, feature_dim=512):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, feature_dim)
        )
    def forward(self, x):
        return x + 0.1 * self.net(x)

class Discriminator(nn.Module):
    """判别特征是来自真实的 (Drug, RNA) 组合还是生成的对抗样本"""
    def __init__(self, feature_dim=512):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x)

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """计算 WGAN-GP 梯度惩罚"""
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones((real_samples.size(0), 1), device=device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgan_gp_step(model, netG, netD, optimizer_G, optimizer_D, adv_gen, batch, device, lambda_gp=10.0):
    """
    使用 AdversarialGenerator 生成的真实对抗样本辅助 WGAN-GP 训练，
    整个过程不更新 model (包含 predictor) 的参数。
    """
    netG.train()
    netD.train()
    
    tok = batch["tok"].to(device)
    rna = batch["rna"].to(device)
    label = batch["label"].to(device)

    # 1. 提取原始样本在模态融合后的特征表示向量 Real Feature (B, 512)
    with torch.no_grad():
        c_repr_real = model.get_emb(tok, rna, t=0)  # Real Features

    # 2. 生成连续空间的对抗样本 (adv_emb, adv_rna)
    # 注意：此处不能放在 torch.no_grad() 内，因为生成对抗样本需要计算 PGD 梯度
    adv_emb, adv_rna, mask = adv_gen.generate(tok, rna, label, device)

    # 3. 提取对抗样本经由 Encoder 和 PMOE 后的特征向量作为 Fake Feature
    # 提取特征时再次使用 torch.no_grad()，确保对抗样本特征提取过程不会更新 model 参数
    with torch.no_grad():
        x, _ = model.drug_encoder.encoder(adv_emb, mask, mask_attn_embed=None, t=0)
        drug_repr_adv = model.drug_encoder.fc(x[:, 0])
        cell_repr_adv, _ = model.pgim(adv_rna)
        cell_repr_adv = model.cell_encoder(cell_repr_adv)
        
        c_repr_adv = torch.cat([drug_repr_adv, cell_repr_adv], dim=1)
        c_repr_adv = c_repr_adv + model.pmoe(c_repr_adv, t=0) # 对抗样本的特征向量

    # ---------------------
    #  训练 Discriminator
    # ---------------------
    optimizer_D.zero_grad()
    
    fake_repr = netG(c_repr_adv).detach()
    
    real_validity = netD(c_repr_real)
    fake_validity = netD(fake_repr)
    
    gp = compute_gradient_penalty(netD, c_repr_real, fake_repr, device)
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
    
    d_loss.backward()
    optimizer_D.step()

    # ---------------------
    #  训练 Generator
    # ---------------------
    optimizer_G.zero_grad()
    
    gen_repr = netG(c_repr_adv)
    gen_validity = netD(gen_repr)
    g_loss = -torch.mean(gen_validity)
    
    g_loss.backward()
    optimizer_G.step()

    return d_loss.item(), g_loss.item()
