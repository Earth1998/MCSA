import numpy as np
from nltk.tokenize import MWETokenizer
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
import pandas as pd
import os
import sys
import datetime
from log_tool import Logger
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import timeit
import scipy
from copy import deepcopy
from itertools import chain
from rdkit import Chem
from drug_vae import *
import utils
from utils import *


def train(model,dataloader,n_epochs, task_id=0):
    start_time = timeit.default_timer()
    loss_lst=[]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.01, verbose=True)
    if model.epoch==0:
        epoch_loss, correlation, rmse, spcc, r2=test(model,teloader[0])
        loss_lst.append(epoch_loss)
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss,5),'correlation:', round(correlation,4),'rmse:', round(rmse,4),'spcc:', round(spcc,5),'r2:', round(r2, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss1, correlation1, rmse1, spcc1, r21=test(model,teloader[1])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss1,5),'correlation:', round(correlation1,4),'rmse:', round(rmse1,4),'spcc:', round(spcc1,5),'r2:', round(r21, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss2, correlation2, rmse2, spcc2, r22=test(model,teloader[2])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss2,5),'correlation:', round(correlation2,4),'rmse:', round(rmse2,4),'spcc:', round(spcc2,5),'r2:', round(r22, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss3, correlation3, rmse3, spcc3, r23=test(model,teloader[3])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss3,5),'correlation:', round(correlation3,4),'rmse:', round(rmse3,4),'spcc:', round(spcc3,5),'r2:', round(r23, 5),'time:',int((time-start_time)/60),'min')
        
        epoch_loss4, correlation4, rmse4, spcc4, r24=test(model,teloader[4])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss4,5),'correlation:', round(correlation4,4),'rmse:', round(rmse4,4),'spcc:', round(spcc4,5),'r2:', round(r24, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss5, correlation5, rmse5, spcc5, r25=test(model,teloader[5])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss5,5),'correlation:', round(correlation5,4),'rmse:', round(rmse5,4),'spcc:', round(spcc5,5),'r2:', round(r25, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss6, correlation6, rmse6, spcc6, r26=test(model,teloader[6])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss6,5),'correlation:', round(correlation6,4),'rmse:', round(rmse6,4),'spcc:', round(spcc6,5),'r2:', round(r26, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss7, correlation7, rmse7, spcc7, r27=test(model,teloader[7])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss7,5),'correlation:', round(correlation7,4),'rmse:', round(rmse7,4),'spcc:', round(spcc7,5),'r2:', round(r27, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss8, correlation8, rmse8, spcc8, r28=test(model,teloader[8])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss8,5),'correlation:', round(correlation8,4),'rmse:', round(rmse8,4),'spcc:', round(spcc8,5),'r2:', round(r28, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss9, correlation9, rmse9, spcc9, r29=test(model,teloader[9])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss9,5),'correlation:', round(correlation9,4),'rmse:', round(rmse9,4),'spcc:', round(spcc9,5),'r2:', round(r29, 5),'time:',int((time-start_time)/60),'min')


    for i in range(n_epochs):
        model.train()
        
        for batch_id, (rna,drug,target) in enumerate(dataloader):
            optimizer.zero_grad()
            result=model(rna,drug)
            output=result[0]
            loss=torch.nn.functional.mse_loss(output,target)
            loss.backward()
            optimizer.step()
            if batch_id%10 ==0: print('batch:',batch_id, 'train loss:',round(loss.item(),3))
        model.epoch+=1
        epoch_loss, correlation, rmse, spcc, r2=test(model,teloader[0])
        # scheduler.step(epoch_loss)
        # creating checkpoints by saving model weights once per epoch:
        # model.save(data_path +'/cl_ewc_10t_3_task{}'.format(task_id))
        loss_lst.append(epoch_loss)
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss,5),'correlation:', round(correlation,4),'rmse:', round(rmse,4),'spcc:', round(spcc,5),'r2:', round(r2, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss1, correlation1, rmse1, spcc1, r21=test(model,teloader[1])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss1,5),'correlation:', round(correlation1,4),'rmse:', round(rmse1,4),'spcc:', round(spcc1,5),'r2:', round(r21, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss2, correlation2, rmse2, spcc2, r22=test(model,teloader[2])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss2,5),'correlation:', round(correlation2,4),'rmse:', round(rmse2,4),'spcc:', round(spcc2,5),'r2:', round(r22, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss3, correlation3, rmse3, spcc3, r23=test(model,teloader[3])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss3,5),'correlation:', round(correlation3,4),'rmse:', round(rmse3,4),'spcc:', round(spcc3,5),'r2:', round(r23, 5),'time:',int((time-start_time)/60),'min')

        epoch_loss4, correlation4, rmse4, spcc4, r24=test(model,teloader[4])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss4,5),'correlation:', round(correlation4,4),'rmse:', round(rmse4,4),'spcc:', round(spcc4,5),'r2:', round(r24, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss5, correlation5, rmse5, spcc5, r25=test(model,teloader[5])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss5,5),'correlation:', round(correlation5,4),'rmse:', round(rmse5,4),'spcc:', round(spcc5,5),'r2:', round(r25, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss6, correlation6, rmse6, spcc6, r26=test(model,teloader[6])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss6,5),'correlation:', round(correlation6,4),'rmse:', round(rmse6,4),'spcc:', round(spcc6,5),'r2:', round(r26, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss7, correlation7, rmse7, spcc7, r27=test(model,teloader[7])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss7,5),'correlation:', round(correlation7,4),'rmse:', round(rmse7,4),'spcc:', round(spcc7,5),'r2:', round(r27, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss8, correlation8, rmse8, spcc8, r28=test(model,teloader[8])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss8,5),'correlation:', round(correlation8,4),'rmse:', round(rmse8,4),'spcc:', round(spcc8,5),'r2:', round(r28, 5),'time:',int((time-start_time)/60),'min')
        epoch_loss9, correlation9, rmse9, spcc9, r29=test(model,teloader[9])
        time = timeit.default_timer()
        print('epochs:', model.epoch,'test loss:',round(epoch_loss9,5),'correlation:', round(correlation9,4),'rmse:', round(rmse9,4),'spcc:', round(spcc9,5),'r2:', round(r29, 5),'time:',int((time-start_time)/60),'min')

    model.set_old_param()
    set_fisher(model, dataloader, t=0)


# test function to evaluate model performance on test set
def test(model,dataloader):
    model.eval()
    avg_loss=0
    avg_correlation=0
    avg_rmse=0
    avg_spcc=0
    avg_r2=0
    
    with torch.no_grad():
        for batch_id, (rna,drug,target) in enumerate(dataloader):
            result=model(rna,drug)
            output=result[0]
            loss=torch.nn.functional.mse_loss(output,target).item()
            correlation=scipy.stats.pearsonr(np.squeeze(output.cpu().numpy()), np.squeeze(target.cpu().numpy()))[0]
            spcc = scipy.stats.spearmanr(np.squeeze(output.cpu().numpy()), np.squeeze(target.cpu().numpy()))[0]
            rmse = np.sqrt(mean_squared_error(np.squeeze(target.cpu().numpy()), np.squeeze(output.cpu().numpy())))
            r2 = r2_score(np.squeeze(target.cpu().numpy()), np.squeeze(output.cpu().numpy()))
            avg_loss=(loss+batch_id*avg_loss)/(batch_id+1)
            avg_correlation=(correlation+batch_id*avg_correlation)/(batch_id+1)
            avg_spcc = (spcc+batch_id*avg_spcc)/(batch_id+1)
            avg_rmse = (rmse+batch_id*avg_rmse)/(batch_id+1)
            avg_r2 = (r2+batch_id*avg_r2)/(batch_id+1)
            #if batch_id==10: break
    return avg_loss, avg_correlation, avg_rmse, avg_spcc, avg_r2
