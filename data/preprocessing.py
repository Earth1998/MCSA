import numpy as np
from nltk.tokenize import MWETokenizer
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import pandas as pd
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import timeit
import scipy
from copy import deepcopy


data_path = './data'

def filter_with_uqstd(df, k=1000):
    uq_perc = df.apply(lambda col: len(col.unique())) / df.shape[0]
    stds = df.std()
    features = (uq_perc * stds).nlargest(k).index.tolist()
    result = df[features]
    return result


def check_nan(df):
    for columname in df.columns:
        if df[columname].count()  != len(df):
            loc = df[columname][df[columname].isnull().values == True].index.tolist()
            print('col name："{}",the{}th row has NA!'.format(columname, loc))


common_gene_gdsc_ccle_tcga_df = pd.read_csv(data_path + '/common_gene_gdsc_ccle_tcga.csv', index_col=0)
common_gene_gdsc_ccle_tcga_pdtc_df = pd.read_csv(data_path + '/common_gene_gdsc_ccle_tcga_pdtc.csv', index_col=0)

print(common_gene_gdsc_ccle_tcga_df)
print(common_gene_gdsc_ccle_tcga_pdtc_df)

common_gene_gdsc_ccle_tcga = common_gene_gdsc_ccle_tcga_df['Gene'].to_list()
common_gene_gdsc_ccle_tcga_pdtc = common_gene_gdsc_ccle_tcga_pdtc_df['Gene'].to_list()


pdtc_rna = pd.read_csv(data_path + '/ExpressionModels.txt', sep='\t', index_col=0)
print(pdtc_rna)
pdtc_rna = pdtc_rna[~pdtc_rna.index.isna()]
print(pdtc_rna)
pdtc_rna = pdtc_rna.fillna(0)
print('check nan', '-' * 128)
check_nan(pdtc_rna)
print('check nan', '-' * 128)

pdtc_rna.index.name = None
print(pdtc_rna)

pdtc_rna = pdtc_rna.T
print(pdtc_rna)


tcga_gex_feature_df = pd.read_csv(data_path + '/TCGA_gene_expression_TPM_gm_tumor.csv', index_col=0)
print(tcga_gex_feature_df)

print('check nan', '-' * 128)
check_nan(tcga_gex_feature_df)
print('check nan', '-' * 128)

tcga_gex_feature_df = np.log2(tcga_gex_feature_df + 1)
print(tcga_gex_feature_df)

gdsc_rna = pd.read_table(data_path + '/Cell_line_RMA_proc_basalExp.txt')
ccle_rna = pd.read_csv(data_path + '/OmicsExpressionProteinCodingGenesTPMLogp1.csv', index_col=0)

print(gdsc_rna)
print(ccle_rna)

gdsc_rna = gdsc_rna.dropna(subset=['GENE_SYMBOLS'])
print(gdsc_rna)

ccle_gene = ccle_rna.columns.to_list()
ccle_gene = [gene.split(' (')[0] for gene in ccle_gene]
ccle_rna.columns = ccle_gene
print(ccle_rna)

pdtc_rna = pdtc_rna[common_gene_gdsc_ccle_tcga_pdtc]
gdsc_rna = gdsc_rna[gdsc_rna['GENE_SYMBOLS'].isin(common_gene_gdsc_ccle_tcga_pdtc)]
ccle_rna = ccle_rna[common_gene_gdsc_ccle_tcga_pdtc]
tcga_gex_feature_df = tcga_gex_feature_df[common_gene_gdsc_ccle_tcga_pdtc]

print(pdtc_rna)
print(gdsc_rna)
print(ccle_rna)
print(tcga_gex_feature_df)

pdtc_rna_copy = deepcopy(pdtc_rna)
gdsc_rna_copy = deepcopy(gdsc_rna)
ccle_rna_copy = deepcopy(ccle_rna)
tcga_rna_copy = deepcopy(tcga_gex_feature_df)

gdsc_rna_copy.index = gdsc_rna_copy['GENE_SYMBOLS'].to_list()
print(gdsc_rna_copy)

gdsc_rna_copy_symbols = deepcopy(gdsc_rna_copy)

gdsc_rna_copy = gdsc_rna_copy.drop('GENE_SYMBOLS', axis=1)
gdsc_rna_copy = gdsc_rna_copy.drop('GENE_title', axis=1)
print(gdsc_rna_copy)

gdsc_rna_copy = gdsc_rna_copy.T
print(gdsc_rna_copy)

print(pdtc_rna_copy)
print(ccle_rna_copy)
print(tcga_rna_copy)


pdtc_rna_copy_0 = deepcopy(pdtc_rna_copy)
gdsc_rna_copy_0 = deepcopy(gdsc_rna_copy)
ccle_rna_copy_0 = deepcopy(ccle_rna_copy)
tcga_rna_copy_0 = deepcopy(tcga_rna_copy)

gdsc_rna_copy_0 = gdsc_rna_copy_0[common_gene_gdsc_ccle_tcga_pdtc]
# pdtc_rna_copy_0.to_csv(data_path + '/pdtc_rna_common_gctp_14390.csv')
# gdsc_rna_copy_0.to_csv(data_path + '/gdsc_rna_common_gctp_14390.csv')
# ccle_rna_copy_0.to_csv(data_path + '/ccle_rna_common_gctp_14390.csv')
# tcga_rna_copy_0.to_csv(data_path + '/tcga_rna_common_gctp_14390.csv')

print('14390 df', '-'*128)
print(pdtc_rna_copy_0)
print(gdsc_rna_copy_0)
print(ccle_rna_copy_0)
print(tcga_rna_copy_0)
print('14390 df', '-'*128)

pdtc_rna_copy = filter_with_uqstd(pdtc_rna_copy, k=2000)
gdsc_rna_copy = filter_with_uqstd(gdsc_rna_copy, k=2000)
ccle_rna_copy = filter_with_uqstd(ccle_rna_copy, k=2000)
tcga_rna_copy = filter_with_uqstd(tcga_rna_copy, k=2000)

print('uq df', '-'*128)
print(pdtc_rna_copy)
print(gdsc_rna_copy)
print(ccle_rna_copy)
print(tcga_rna_copy)
print('uq df', '-'*128)

pdtc_uq_gene = pdtc_rna_copy.columns.to_list()
gdsc_uq_gene = gdsc_rna_copy.columns.to_list()
ccle_uq_gene = ccle_rna_copy.columns.to_list()
tcga_uq_gene = tcga_rna_copy.columns.to_list()

print(len(pdtc_uq_gene))
print(len(gdsc_uq_gene))
print(len(ccle_uq_gene))
print(len(tcga_uq_gene))

uq_gene = pdtc_uq_gene + gdsc_uq_gene + ccle_uq_gene + tcga_uq_gene

uq_gene = list(set(uq_gene))

uq_gene = sorted(uq_gene)

print(len(uq_gene))

pdtc_uq_rna = pdtc_rna_copy_0[uq_gene]
gdsc_uq_rna = gdsc_rna_copy_0[uq_gene]
ccle_uq_rna = ccle_rna_copy_0[uq_gene]
tcga_uq_rna = tcga_rna_copy_0[uq_gene]

print(pdtc_uq_rna)
print(gdsc_uq_rna)
print(ccle_uq_rna)
print(tcga_uq_rna)

# Normalization Save ----------------------------------------------------------------------------------------------------------------

# scaler = StandardScaler()

# pdtc_uq_rna_normalized = pd.DataFrame(scaler.fit_transform(pdtc_uq_rna), columns=pdtc_uq_rna.columns, index=pdtc_uq_rna.index)
# gdsc_uq_rna_normalized = pd.DataFrame(scaler.fit_transform(gdsc_uq_rna), columns=gdsc_uq_rna.columns, index=gdsc_uq_rna.index)
# ccle_uq_rna_normalized = pd.DataFrame(scaler.fit_transform(ccle_uq_rna), columns=ccle_uq_rna.columns, index=ccle_uq_rna.index)
# tcga_uq_rna_normalized = pd.DataFrame(scaler.fit_transform(tcga_uq_rna), columns=tcga_uq_rna.columns, index=tcga_uq_rna.index)

# print('normalized uq df', '-'*128)
# print(pdtc_uq_rna_normalized)
# print(gdsc_uq_rna_normalized)
# print(ccle_uq_rna_normalized)
# print(tcga_uq_rna_normalized)
# print('normalized uq df', '-'*128)

# print(gdsc_rna_copy_symbols)

# gdsc_rna_copy_symbols = gdsc_rna_copy_symbols.loc[gdsc_uq_rna_normalized.columns.to_list(), :]

# print(gdsc_rna_copy_symbols)

# gdsc_rna_copy_symbols.iloc[:, 2:] = gdsc_uq_rna_normalized.T.values

# print(gdsc_rna_copy_symbols)

# gdsc_rna_copy_symbols = gdsc_rna_copy_symbols.drop('GENE_SYMBOLS', axis=1)

# print(gdsc_rna_copy_symbols)

# gdsc_rna_copy_symbols.index.name = 'GENE_SYMBOLS'

# print(gdsc_rna_copy_symbols)

# pdtc_uq_rna_normalized.to_csv(data_path + '/pdtc_rna_uq_gctpm_2000.csv')
# gdsc_rna_copy_symbols.to_csv(data_path + '/gdsc_rna_uq_gctpm_2000.csv')
# ccle_uq_rna_normalized.to_csv(data_path + '/ccle_rna_uq_gctpm_2000.csv')
# tcga_uq_rna_normalized.to_csv(data_path + '/tcga_rna_uq_gctpm_2000.csv')

# Normalization Save ----------------------------------------------------------------------------------------------------------------

# Raw Save --------------------------------------------------------------------------------------------------------------------------

print(gdsc_rna_copy_symbols)

gdsc_rna_copy_symbols = gdsc_rna_copy_symbols.loc[gdsc_uq_rna.columns.to_list(), :]

print(gdsc_rna_copy_symbols)

gdsc_rna_copy_symbols = gdsc_rna_copy_symbols.drop('GENE_SYMBOLS', axis=1)

print(gdsc_rna_copy_symbols)

gdsc_rna_copy_symbols.index.name = 'GENE_SYMBOLS'

print(gdsc_rna_copy_symbols)

pdtc_uq_rna.to_csv(data_path + '/pdtc_rna_uq_gctpm_2000_raw.csv')
gdsc_rna_copy_symbols.to_csv(data_path + '/gdsc_rna_uq_gctpm_2000_raw.csv')
ccle_uq_rna.to_csv(data_path + '/ccle_rna_uq_gctpm_2000_raw.csv')
tcga_uq_rna.to_csv(data_path + '/tcga_rna_uq_gctpm_2000_raw.csv')
