# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:42:41 2025

@author: wangl
"""

# %%
# -*- coding: utf-8 -*-

"""
Created on Mon Nov 25 17:19:04 2024

@author: wangl
"""

from src.lfgp_withoutO_biased import LFGP
from src.GTIC import gtic
from src.multispa import multispa_fit_predict
from src.getdata import getdata
from src.getdata_biased import getdata_biased
from src.eigenInfer import _hq_and_label_infer
from src.peera import peerA
import numpy as np
import warnings
import numpy_indexed as npi
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
import seaborn as sns
import itertools
import numpy as np
import time

start = time.perf_counter()
warnings.filterwarnings('ignore')

n_task=200
n_worker=200
n_task_groups=5
n_worker_groups=10    #used for m = n_worker // n_worker_groups for number of hq workers
task_accuracy=[]
task_accuracy_ds=[]
task_accuracy_MV= []
task_accuracy_MV_HQ = []
task_accuracy_gtic = []
task_accuracy_multispa = []
task_accuracy_glad = []
maxiter = 50
acc_k = []
l=[]

eigen_ex= 1
DS_ex = 1
MV_HQ_ex= 0
MV_ex = 0
GTIC_ex = 0
Multispa_ex = 0
GLAD_ex = 0
'''
ratios = [1/2, 1/3, 1/4]
combinations = [
    (hq, bias)
    for hq, bias in itertools.product(ratios, ratios)
]

for hq_ratio, bias_ratio in combinations:
    print(f"\n=== hq_ratio={hq_ratio:.3f}, bias_ratio={bias_ratio:.3f} ===")
'''
for i in range(2):
    
    #np.random.seed(i)
    '''
    rating, label, worker_label, R_obs, task_lf, worker_lf = getdata( n_task = n_task,
     n_worker = n_worker,
     n_task_groups = n_task_groups,  
     k = 5,
     sigma = 0.1,
     obs_prob=0.3,
     noise_ratio = 0.95,   # fraction of low-quality workers per group
     n_classes = n_task_groups)
    '''


    rating, label, worker_label, R_obs, task_lf, worker_lf = getdata_biased(
        n_task = n_task,
        n_worker = n_worker,
        n_task_groups = n_task_groups,
        k = 3,
        sigma = 1.0,
        obs_prob = 0.3,
        hq_ratio = 1/4,
        bias_ratio = 1/2,
        delta = 1,
        n_classes = 5
    )


    #All workers on one task group agreement
    hq_workers_by_group = []  # will store inferred HQ workers for each group
    results = []
    # optional: collect metrics across k & groups

    if eigen_ex:
        model = LFGP(lf_dim=n_task_groups, n_worker_group=n_task_groups, lambda1 = 1, lambda2_0 = 1, lambda2_1 = 2)
        model._prescreen(rating)
        
        _, task_id = np.unique(rating[:, 0], return_inverse=True)
        _, worker_id = np.unique(rating[:, 1], return_inverse=True)
        
    
        
        A, B, U, V, _ = model._mc_fit(rating, key = label, scheme="ds", epsilon=1e-5, maxiter=maxiter, verbose=0, A = task_lf, B = worker_lf)
        ###############################################
        # After LFGP fit: U is length n_task
        ###############################################
        
        # Ensure U is integer group labels for tasks
        
        pred_group = U.astype(int) 
        task_labeling_acc = model.task_acc(rating, label)
        print("==== TASK GROUPING ACCURACY ====")
        print(task_labeling_acc)
        
        temp_taskAccuracy, task_label_pred, hq_workers_pred = _hq_and_label_infer(pred_group, 
                                R_obs,
                                label,
                                worker_label,
                                n_task, 
                                n_worker,
                                n_task_groups,
                                n_worker_groups,
                                USE_TOP2_EIGEN = True,
                                LABEL_MODE = 'group',
                                verbose = True
                                )
        '''
        U_mv_by_task = model._mc_infer_by_task(rating)
        temp_taskAccuracy = np.mean(U_mv_by_task==label)
        '''
        task_accuracy.append(temp_taskAccuracy)
    if DS_ex:
        pred_label_ds = model._init_task_member_ds(rating)
        task_accuracy_ds.append(np.mean(pred_label_ds[:, 1] == label))
    if MV_HQ_ex:
        #Data generation accuracy with hq workers
        MV_HQ = np.zeros(n_task)
        for t in range(n_task):
            task_t = label[t]
            hq_worker = np.where(worker_label[:, task_t] == 1)[0]        
            task_data = rating[np.isin(rating[:, 1], hq_worker) & (rating[:, 0] == t)]
            labels = task_data[:, 2]
            if len(labels) == 0:
                MV_HQ[t] = -1  # or np.nan, or whatever sentinel makes sense for your use case
            else:
                MV_HQ[t] = mode(labels, axis=None).mode.item()
            
        task_accuracy_MV_HQ.append(np.mean(MV_HQ == label))
    if MV_ex:
        #Data generation accuracy with all workers
        
        MV = np.zeros(n_task, dtype=int)
    
        for t in range(n_task):
            task_data = rating[rating[:, 0] == t]
            labels = task_data[:, 2]
            if len(labels) == 0:
                MV[t] = -1
            else:
                MV[t] = mode(labels, axis=None).mode.item()
            
        task_accuracy_MV.append(np.mean(MV == label))
    if GLAD_ex:
        peera = peerA(rating, n_task_groups, n_worker)
        res_glad, _ = peera._GLAD()
        task_accuracy_glad.append(np.mean(res_glad == label))
        
    if Multispa_ex:    
        res_multispa = multispa_fit_predict(rating, K=n_task_groups, assume_triplets=True)
        task_accuracy_multispa.append(np.mean(res_multispa.y_hat == label))
        
    if GTIC_ex:    
        res_gtic = gtic(rating, n=n_task, m=n_worker, K=n_task_groups, missing_val=-1)
        task_accuracy_gtic.append(np.mean(res_gtic.y_hat == label))
    
    
if eigen_ex:
    task_accuracy_mean = np.mean(task_accuracy)
    task_accuracy_sd = np.std(task_accuracy)
    results = {
        "Eigen_L2 without eign": {
            "mean": task_accuracy_mean,
            "sd": task_accuracy_sd
        }}
    print(pd.DataFrame.from_dict(results, orient="index"))
if DS_ex:
    task_accuracy_mean_ds = np.mean(task_accuracy_ds)
    task_accuracy_sd_ds = np.std(task_accuracy_ds)
    results = {
        "DS": {
            "mean": task_accuracy_mean_ds,
            "sd": task_accuracy_sd_ds
        }}
    print(pd.DataFrame.from_dict(results, orient="index"))
if MV_HQ_ex:
    task_accuracy_mean_MV_HQ = np.mean(task_accuracy_MV_HQ)
    task_accuracy_sd_MV_HQ = np.std(task_accuracy_MV_HQ)
    results = {
        "MV_HQ": {
            "mean": task_accuracy_mean_MV_HQ,
            "sd": task_accuracy_sd_MV_HQ
        }}
    print(pd.DataFrame.from_dict(results, orient="index"))
if MV_ex:
    task_accuracy_mean_MV = np.mean(task_accuracy_MV)
    task_accuracy_sd_MV = np.std(task_accuracy_MV)
    results = {
        "MV": {
            "mean": task_accuracy_mean_MV,
            "sd": task_accuracy_sd_MV
        }}
    print(pd.DataFrame.from_dict(results, orient="index"))
if GTIC_ex:
    task_accuracy_mean_gtic = np.mean(task_accuracy_gtic)
    task_accuracy_sd_gtic = np.std(task_accuracy_gtic)
    results = {
        "GTIC": {
            "mean": task_accuracy_mean_gtic,
            "sd": task_accuracy_sd_gtic
        }}
    print(pd.DataFrame.from_dict(results, orient="index"))
if Multispa_ex:
    task_accuracy_mean_multispa = np.mean(task_accuracy_multispa)
    task_accuracy_sd_multisp = np.std(task_accuracy_multispa)
    results = {
        "MultiSPA": {
            "mean": task_accuracy_mean_multispa,
            "sd": task_accuracy_sd_multisp
        }}
    print(pd.DataFrame.from_dict(results, orient="index"))
if GLAD_ex:
    task_accuracy_mean_glad = np.mean(task_accuracy_glad)
    task_accuracy_sd_glad = np.std(task_accuracy_glad)
    results = {
        "GLAD": {
            "mean": task_accuracy_mean_glad,
            "sd": task_accuracy_sd_glad
        }}
    print(pd.DataFrame.from_dict(results, orient="index"))

end = time.perf_counter()
runtime = end - start

print("Runtime:", runtime)



    





    
    
    
    

    
