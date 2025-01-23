import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import pandas as pd
import os
import sys
import random
from sklearn.model_selection import train_test_split
import scipy


def split(df, col, n_split, shuffle=False):
    if n_split < 2:
        raise ValueError('Only support n_split more than 1')
    print(df[col].value_counts(dropna=False))
    col_list = df[col].value_counts(dropna=False).index.to_list()
    if shuffle:
        random.shuffle(col_list)
    l = len(col_list)
    step = int(l / n_split)
    split_col_list = [col_list[i:i+step] for i in range(0, l, step)]
    if len(split_col_list) > n_split:
        split_col_list[-2].extend(split_col_list[-1])
        split_col_list.pop()
    return split_col_list



def split_data(df, col, ratio, split_list):
    sub_df = df[df[col].isin(split_list)]
    train_data, test_data = train_test_split(sub_df, test_size=1-ratio, stratify=sub_df['DRUG_ID'])
    return train_data, test_data


# GDSC_data=pd.read_excel(data_path + '/GDSC2_fitted_dose_response_25Feb20.xlsx')
# drug_data=pd.read_csv(data_path + '/SMILEinchi.csv')
GDSC_data=pd.read_excel(data_path + '/GDSC1_fitted_dose_response_27Oct23.xlsx')
drug_data=pd.read_csv(data_path + '/smile_inchi.csv')
# gene_data=pd.read_table(data_path + '/Cell_line_RMA_proc_basalExp.txt')
gene_data = pd.read_csv(data_path + '/gdsc_rna_uq2000.csv')
# gene_data=pd.read_excel(data_path + '/Cell_line_RMA_uq2000.xlsx')

gene_names=list(gene_data.columns)
gene_names.pop(0)
gene_names.pop(0)
for i in range(len(gene_names)):
    gene_names[i] = gene_names[i].split('A.')[1]
# gene_names = [int(float(i)) for i in gene_names]

GDSC_data = GDSC_data[GDSC_data['COSMIC_ID'].astype('str').isin(gene_names)]
# GDSC_data = GDSC_data[GDSC_data['DRUG_ID'].isin(drug_data['drug_id'])]
GDSC_data = GDSC_data[GDSC_data['DRUG_ID'].isin(drug_data['Drug Id'])]
print(GDSC_data.shape)
split_col = 'DRUG_ID'
n_split = 10
split_list = split(GDSC_data, split_col, n_split, shuffle=True)
print(split_list)

GDSC_train_data, GDSC_test_data = split_data(GDSC_data, split_col, 0.9, split_list[0])


class CombinedDataset(torch.utils.data.Dataset):

    def __init__(self,gene_pca=False, split_col=None, split_list=None, load_df=None):

        # loading files
        # self.gene_data=pd.read_table(data_path + '/Cell_line_RMA_proc_basalExp.txt')
        self.gene_data = pd.read_csv(data_path + '/gdsc_rna_uq2000.csv')
        # self.gene_data=pd.read_excel(data_path + '/Cell_line_RMA_uq2000.xlsx')
        if load_df is None:
            # GDSC_data=pd.read_excel(data_path + '/GDSC2_fitted_dose_response_25Feb20.xlsx')
            GDSC_data=pd.read_excel(data_path + '/GDSC1_fitted_dose_response_27Oct23.xlsx')
        else:
            GDSC_data = load_df
        self.drug_data=pd.read_pickle(data_path +'/token_eos1.pkl')
        # self.drug_data=pd.read_pickle(data_path +'/token1.pkl')

        if split_col is not None:
            GDSC_data = GDSC_data[GDSC_data[split_col].isin(split_list)]
        # GDSC_data = GDSC_data[GDSC_data['COSMIC_ID'].isin(gene_names[0:300])]

        # reduce size of gene_data file
        self.gene_names=list(self.gene_data.columns)
        self.gene_names.pop(0)
        self.gene_names.pop(0)
        self.gene_data=self.gene_data.T
        self.gene_data=self.gene_data.drop(['GENE_SYMBOLS','GENE_title'])
        self.gene_data=self.gene_data.astype('float32')

        if gene_pca:
            pca = PCA(n_components=1018)
            pca.fit(self.gene_data)
            self.gene_data=pca.transform(self.gene_data)

        self.gene_data=self.gene_data.T
        self.gene_data=pd.DataFrame(self.gene_data)
        self.gene_data=self.gene_data.set_axis(self.gene_names,axis=1)
        # self.gene_data = self.gene_data.iloc[:, 0:300]

        COSMIC_lst0= list(GDSC_data['COSMIC_ID'])
        drugID_lst0= list(GDSC_data['DRUG_ID'])
        IC50_lst0= list(GDSC_data['LN_IC50'])
        self.COSMIC_lst=[]
        self.drugID_lst=[]
        self.IC50_lst=[]

        # dropping entries with no corresponding gene/drug data
        n_fails=0
        for i in range(len(COSMIC_lst0)):
            try:
                self.gene_data['DATA.'+str(COSMIC_lst0[i])]
                self.drug_data[drugID_lst0[i]]

                self.COSMIC_lst.append(COSMIC_lst0[i])
                self.drugID_lst.append(drugID_lst0[i])
                self.IC50_lst.append(IC50_lst0[i])
            except: n_fails+=1
        print('n_fails', n_fails)
        # drug_token=torch.Tensor(self.drug_data.loc['token',self.drugID_lst[0]])

    def __len__(self):
        return(len(self.IC50_lst))

    def __getitem__(self,i):
        gene_expression=torch.Tensor(list(self.gene_data['DATA.'+str(self.COSMIC_lst[i])]))
        drug_token=torch.Tensor(self.drug_data.loc['token',self.drugID_lst[i]])
        IC50_value=torch.Tensor([self.IC50_lst[i]])
        return gene_expression, drug_token, IC50_value


# gene_pca reduces rna input dim from 17737 to 1018, somewhat faster. Layers in the model have to be adjusted if changed.
train_set=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[0], load_df=GDSC_train_data)
test_set=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[0], load_df=GDSC_test_data)

GDSC_train_data, GDSC_test_data =split_data(GDSC_data, split_col, 0.9, split_list[1])
train_set1=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[1], load_df=GDSC_train_data)
test_set1=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[1], load_df=GDSC_test_data)

GDSC_train_data, GDSC_test_data =split_data(GDSC_data, split_col, 0.9, split_list[2])
train_set2=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[2], load_df=GDSC_train_data)
test_set2=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[2], load_df=GDSC_test_data)

GDSC_train_data, GDSC_test_data =split_data(GDSC_data, split_col, 0.9, split_list[3])
train_set3=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[3], load_df=GDSC_train_data)
test_set3=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[3], load_df=GDSC_test_data)

GDSC_train_data, GDSC_test_data =split_data(GDSC_data, split_col, 0.9, split_list[4])
train_set4=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[4], load_df=GDSC_train_data)
test_set4=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[4], load_df=GDSC_test_data)

GDSC_train_data, GDSC_test_data =split_data(GDSC_data, split_col, 0.9, split_list[5])
train_set5=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[5], load_df=GDSC_train_data)
test_set5=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[5], load_df=GDSC_test_data)

GDSC_train_data, GDSC_test_data =split_data(GDSC_data, split_col, 0.9, split_list[6])
train_set6=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[6], load_df=GDSC_train_data)
test_set6=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[6], load_df=GDSC_test_data)

GDSC_train_data, GDSC_test_data =split_data(GDSC_data, split_col, 0.9, split_list[7])
train_set7=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[7], load_df=GDSC_train_data)
test_set7=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[7], load_df=GDSC_test_data)

GDSC_train_data, GDSC_test_data =split_data(GDSC_data, split_col, 0.9, split_list[8])
train_set8=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[8], load_df=GDSC_train_data)
test_set8=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[8], load_df=GDSC_test_data)

GDSC_train_data, GDSC_test_data =split_data(GDSC_data, split_col, 0.9, split_list[9])
train_set9=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[9], load_df=GDSC_train_data)
test_set9=CombinedDataset(gene_pca=False, split_col=split_col, split_list=split_list[9], load_df=GDSC_test_data)
