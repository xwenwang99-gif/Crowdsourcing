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

from src.lfgp_withoutO import LFGP
from src.GTIC import gtic
from src.multispa import multispa_fit_predict
from src.getdata import getdata
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
from google.colab import drive
import numpy as np
import time

start = time.perf_counter()
warnings.filterwarnings('ignore')

n_task=1000
n_worker=1000
n_task_groups=5
n_worker_groups=10
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


for i in range(5):
    
    ##rating, label = getdata1(scenario="hetero", seed=seed)
    ##getdata1
    np.random.seed(i)
    
    rating, label, worker_label, R_obs = getdata(n_task, 
    n_worker,
    n_task_groups,
    n_worker_groups,    
    k=2,
    sigma= 1.0,
    obs_prob=0.3,
    noise_group= 5)


    #All workers on one task group agreement
    hq_workers_by_group = []  # will store inferred HQ workers for each group
    results = []
    # optional: collect metrics across k & groups


    model = LFGP(lf_dim=5, n_worker_group=10, lambda1 = 1, lambda2_0 = 1, lambda2_1 = 2)
    model._prescreen(rating)
    
    _, task_id = np.unique(rating[:, 0], return_inverse=True)
    _, worker_id = np.unique(rating[:, 1], return_inverse=True)
    

    
    A, B, U, V, _ = model._mc_fit(rating, key = label, epsilon=1e-5, maxiter=maxiter, verbose=0)
    ###############################################
    # After LFGP fit: U is length n_task
    ###############################################
    
    # Ensure U is integer group labels for tasks
    
    pred_group = U.astype(int) 
    
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
                            verbose = False
                            )
    
    task_accuracy.append(temp_taskAccuracy)
    
    pred_label_ds = model._init_task_member_ds(rating)
    task_accuracy_ds.append(np.mean(pred_label_ds[:, 1] == label))
    
    #Data generation accuracy with hq workers
    MV_HQ = np.zeros(n_task)
    for t in range(n_task):
        task_t = label[t]
        hq_worker = np.where(worker_label[:, task_t] == 1)[0]        
        task_data = rating[np.isin(rating[:, 1], hq_worker) & (rating[:, 0] == t)]
        labels = task_data[:, 2]
        MV_HQ[t] = mode(labels, axis=None).mode
        
    task_accuracy_MV_HQ.append(np.mean(MV_HQ == label))
    
    #Data generation accuracy with all workers
    
    MV = np.zeros(n_task, dtype=int)

    for t in range(n_task):
        task_data = rating[rating[:, 0] == t]
        labels = task_data[:, 2]
        MV[t] = mode(labels, axis=None).mode
        
    task_accuracy_MV.append(np.mean(MV == label))
    
    peera = peerA(rating, n_task_groups, n_worker)
    res_glad, _ = peera._GLAD()
    task_accuracy_glad.append(np.mean(res_glad == label))
    
    res_multispa = multispa_fit_predict(rating, K=n_task_groups, assume_triplets=True)
    task_accuracy_multispa.append(np.mean(res_multispa.y_hat == label))
    
    res_gtic = gtic(rating, n=n_task, m=n_worker, K=n_task_groups, missing_val=-1)
    task_accuracy_gtic.append(np.mean(res_gtic.y_hat == label))
    
    

task_accuracy_mean = np.mean(task_accuracy)
task_accuracy_sd = np.std(task_accuracy)
task_accuracy_mean_ds = np.mean(task_accuracy_ds)
task_accuracy_sd_ds = np.std(task_accuracy_ds)
task_accuracy_mean_MV_HQ = np.mean(task_accuracy_MV_HQ)
task_accuracy_sd_MV_HQ = np.std(task_accuracy_MV_HQ)
task_accuracy_mean_MV = np.mean(task_accuracy_MV)
task_accuracy_sd_MV = np.std(task_accuracy_MV)
task_accuracy_mean_gtic = np.mean(task_accuracy_gtic)
task_accuracy_sd_gtic = np.std(task_accuracy_gtic)
task_accuracy_mean_multispa = np.mean(task_accuracy_multispa)
task_accuracy_sd_multisp = np.std(task_accuracy_multispa)
task_accuracy_mean_glad = np.mean(task_accuracy_glad)
task_accuracy_sd_glad = np.std(task_accuracy_glad)

results = {
    "Eigen_L2": {
        "mean": task_accuracy_mean,
        "sd": task_accuracy_sd
    },
    "DS": {
        "mean": task_accuracy_mean_ds,
        "sd": task_accuracy_sd_ds
    },
    "MV_HQ": {
        "mean": task_accuracy_mean_MV_HQ,
        "sd": task_accuracy_sd_MV_HQ
    },
    "MV": {
        "mean": task_accuracy_mean_MV,
        "sd": task_accuracy_sd_MV
    },
    "GTIC": {
        "mean": task_accuracy_mean_gtic,
        "sd": task_accuracy_sd_gtic
    },
    "MultiSPA": {
        "mean": task_accuracy_mean_multispa,
        "sd": task_accuracy_sd_multisp
    },
    "GLAD": {
        "mean": task_accuracy_mean_glad,
        "sd": task_accuracy_sd_glad
    }
}

end = time.perf_counter()
runtime = end - start

drive.mount('/content/drive')
df = pd.DataFrame.from_dict(results, orient="index")
df.to_csv("/content/drive/MyDrive/simulation_results.csv")

print(df)
print("Runtime:", runtime)



    





    
    
    
    

    
