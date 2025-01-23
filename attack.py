import torch
import torch.nn as nn


class Attack(object):
    def __init__(self, old_model, new_model, alpha, loader, proto, device, epochs, 
                 rna_min, rna_max, drug_min, drug_max, target):
        self.old_model = old_model
        self.new_model = new_model
        self.alpha = alpha
        self.loader = loader
        self.device = device
        self.target = target
        self.epochs = epochs
        self.proto = proto
        self.rna_min = rna_min
        self.rna_max = rna_max
        self.drug_min = drug_min
        self.drug_max = drug_max

    def perturb(self, rna, drug, alpha, rna_grad, drug_grad, rna_min, rna_max, drug_min, drug_max):
        rna_prime = rna - (alpha * rna_grad / torch.norm(rna_grad, keepdim=True))
        rna_prime = torch.clamp(rna_prime, rna_min.to(self.device), rna_max.to(self.device))
        drug_prime = drug - (alpha * drug_grad / torch.norm(drug_grad, keepdim=True))
        drug_prime = torch.clamp(drug_prime, drug_min.to(self.device), drug_max.to(self.device))
        return rna_prime, drug_prime
    
    def run(self):
        print(self.old_model.training)
        p_rna, p_drug, p_label = [], [], []
    
        for rna, drug, label in self.loader:
            rna, drug, label = rna.to(self.device), drug.to(self.device), label.to(self.device)
            target = torch.tensor(self.target).expand(len(label)).to(self.device)
            
            for i in range(self.epochs):
                # Adversarial attack requires gradients w.r.t. the data
                rna.requires_grad = True
                drug.requires_grad = True
                feats = self.old_model.encode(rna, drug)
                L = nn.MSELoss()

                loss = L(feats, self.proto.expand(len(label), self.proto.shape[0]).to(self.device))
            
                self.old_model.zero_grad()

                loss.backward()
                rna_grad = rna.grad
                drug_grad = drug.grad
                print(rna_grad)
            
                perturbed_rna, perturbed_drug = self.perturb(rna, drug, self.alpha, rna_grad, drug_grad, self.rna_min, self.rna_max, self.drug_min, self.drug_max)
                rna = perturbed_rna.clone().detach()
                drug = perturbed_drug.clone().detach()

                if i == (self.epochs-1):
                    p_rna.append(perturbed_rna.detach())
                    p_drug.append(perturbed_drug.detach())
                    p_label.append(self.old_model(perturbed_rna, perturbed_drug)[0].detach())

        return torch.cat(p_rna, 0), torch.cat(p_drug, 0), torch.cat(p_label, 0)
