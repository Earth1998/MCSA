import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.epoch=0
        self.device='cpu'
        self.emb=Embedding(2714,59,8)
        self.trafo=Transformer(8)
        self.dropout=nn.Dropout()
        self.relu=torch.nn.ReLU()
        self.lin_1=nn.Linear(2000,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.lin_2=nn.Linear(256,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.class_1=nn.Linear(536,64)
        self.bn3 = nn.BatchNorm1d(64)
        self.class_2=nn.Linear(64,8)
        self.bn4 = nn.BatchNorm1d(8)
        self.class_3=nn.Linear(8,1)
        self.optimizer=torch.optim.AdamW(self.parameters(), lr=0.001)

        self.old_param = {}
        self.fisher=None
        self._protos = []

    
    #     # implementation 1
    #     modules = []
    #     modules.append(
    #         nn.Sequential(
    #             nn.Linear(1018,256),
    #             nn.BatchNorm1d(256),
    #             nn.ReLU()
    #         )
    #     )
    #     modules.append(
    #         nn.Sequential(
    #             nn.Linear(256,64),
    #             nn.BatchNorm1d(64),
    #             nn.ReLU()
    #         )
    #     )

    #     self.rna_encoder = nn.Sequential(*modules)

    #     modules = []
    #     modules.append(Embedding(2713,58,8))
    #     modules.append(Transformer(8))

    #     self.drug_encoder = nn.Sequential(*modules)

    #     modules = []
    #     modules.append(
    #         nn.Sequential(
    #             nn.Linear(528,64),
    #             nn.BatchNorm1d(64),
    #             nn.ReLU()
    #         )
    #     )
    #     modules.append(
    #         nn.Sequential(
    #             nn.Linear(64,8),
    #             nn.BatchNorm1d(8),
    #             nn.ReLU()
    #         )
    #     )

    #     self.decoder = nn.Sequential(*modules)

    #     self.final_layer = nn.Linear(8,1)
    #     self.optimizer=torch.optim.AdamW(self.parameters(), lr=0.001)
    
    # def encode(self, rna, drug):
    #     latent_rna = self.rna_encoder(rna)
    #     latent_drug = self.drug_encoder(drug)
    #     latent_drug = torch.flatten(latent_drug,1)
    #     latent_code = torch.cat((latent_rna,latent_drug),dim=1)

    #     return latent_code
    
    # def decode(self, z):
    #     embed = self.decoder(z)
    #     outputs = self.final_layer(embed)

    #     return outputs
    
    # def forward(self,rna,drug):
    #     z = self.encode(rna,drug)
    #     outputs = self.decode(z)
    #     return outputs
    
    # implementation 2
    def encode(self, rna, drug):
        # rna linear layers
        rna=self.lin_1(rna)
        rna=self.bn1(rna)
        rna=self.relu(rna)
        rna=self.lin_2(rna)
        rna=self.bn2(rna)
        rna=self.relu(rna)
        # drug embedding and transformer layers
        drug=self.emb(drug)
        drug=self.trafo(drug)
        drug=torch.flatten(drug,1)
        # classifier layer
        x=torch.cat((rna,drug),dim=1)

        return x
    
    def decode(self, x):
        x=self.class_1(x)
        x=self.bn3(x)
        x=self.relu(x)
        x=self.class_2(x)
        x=self.bn4(x)
        x=self.relu(x)

        return x
    
    def forward(self,rna,drug):
        z = self.encode(rna,drug)
        x = self.decode(z)
        output=self.class_3(x)
        return [output, z]

    # def forward(self,rna,drug):
    #     # rna linear layers
    #     rna=self.lin_1(rna)
    #     rna=self.bn1(rna)
    #     rna=self.relu(rna)
    #     rna=self.lin_2(rna)
    #     rna=self.bn2(rna)
    #     rna=self.relu(rna)
    #     # drug embedding and transformer layers
    #     drug=self.emb(drug)
    #     drug=self.trafo(drug)
    #     drug=torch.flatten(drug,1)
    #     # classifier layer
    #     x=torch.cat((rna,drug),dim=1)
    #     x=self.class_1(x)
    #     x=self.bn3(x)
    #     x=self.relu(x)
    #     x=self.class_2(x)
    #     x=self.bn4(x)
    #     x=self.relu(x)
    #     output=self.class_3(x)
    #     return output

    def save(self, path):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch=checkpoint['epoch']
    
    def set_old_param(self):
        for n, p in self.named_parameters():
            self.old_param[n] = p.data.clone().detach()
    
    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for name, param in self.named_parameters():
                loss_reg += torch.sum(self.fisher[name] * (self.old_param[name] - param).pow(2)) / 2
        
        return 1000000 * loss_reg
    
    def build_protos(self, t, loader):
        self.eval()
        feats_ = []
        for batch_id, (rna,drug,target) in enumerate(loader):
            with torch.no_grad():
                feats = self.encode(rna, drug)
                feats_.append(feats.detach().cpu())
        feats_ = torch.cat(feats_, dim=0)
        feats_ = feats_.numpy()
        km = KMeans(n_clusters=23, n_init='auto').fit(feats_)
        centers = km.cluster_centers_
        centers = torch.Tensor(centers).to(self.device)
        self._protos.append(centers)
