import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import batched


def pathway_splitter(shuffle=False, batch_size=2):
    gdsc2_rma = pd.read_csv("data/gdsc_csv/gdsc2_rma.csv")
    gdsc2_rma = gdsc2_rma[~gdsc2_rma["PATHWAY_NAME"].isin(["Other", "Other, kinases", "Unclassified", "Chromatin other"])]
    PATHWAY_NAME = gdsc2_rma["PATHWAY_NAME"].value_counts().index.to_list()
    index_list = []
    if shuffle:
        random.shuffle(PATHWAY_NAME)
    for pathway in batched(PATHWAY_NAME, batch_size):
        df = gdsc2_rma[gdsc2_rma["PATHWAY_NAME"].isin(list(pathway))]
        if len(set(df["smiles"])) > 1:
            train_df, temp_df = train_test_split(
                df,
                test_size=0.2,
                stratify=df["smiles"],
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                stratify=temp_df["smiles"],
            )
        else:
            train_df, temp_df = train_test_split(
                df,
                test_size=0.2,
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
            )
        train_idx = train_df.index.to_list()
        val_idx = val_df.index.to_list()
        test_idx = test_df.index.to_list()
        
        index_list.append({
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        })
    
    return index_list
