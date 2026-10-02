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

from src.lfgp_paper import LFGP
from src.getdata import getdata
import numpy as np
import warnings
import numpy_indexed as npi
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
import seaborn as sns
import itertools

import numpy as np

warnings.filterwarnings('ignore')
sigma = 1
k=3
acc = np.zeros(1)
iters=5
maxiter = 50
n_task, n_worker = 200, 400
n_task_groups = 5
n_worker_groups=7
tmp = np.zeros((7, iters))
acc_k = []
acc_MV_HQ = []
acc_MV = []
task_accuracy = []


for i in range(3):
    
    ##rating, label = getdata1(scenario="hetero", seed=seed)
    ##getdata1
    np.random.seed(i)
    
    rating, label, worker_label, R_obs = getdata(n_task, 
    n_worker,
    n_task_groups,
    n_worker_groups,    
    k=3,
    sigma= 1.0,
    obs_prob=0.3,
    noise_group= 5)
    
    #Data generation accuracy with hq workers
    MV_HQ = np.zeros(n_task)
    for t in range(n_task):
        task_t = label[t]
        hq_worker = np.where(worker_label[:, task_t] == 1)[0]        
        task_data = rating[np.isin(rating[:, 1], hq_worker) & (rating[:, 0] == t)]
        labels = task_data[:, 2]
        MV_HQ[t] = mode(labels, axis=None).mode
        
    acc_MV_HQ.append(np.mean(MV_HQ == label))
    
    #Data generation accuracy with all workers
    
    MV = np.zeros(n_task, dtype=int)

    for t in range(n_task):
        task_data = rating[rating[:, 0] == t]
        labels = task_data[:, 2]
        MV[t] = mode(labels, axis=None).mode
        
    acc_MV.append(np.mean(MV == label))


    model = LFGP(lf_dim=5, n_worker_group=5, lambda1=1, lambda2=1)
    model._prescreen(rating)
    
    _, task_id = np.unique(rating[:, 0], return_inverse=True)
    _, worker_id = np.unique(rating[:, 1], return_inverse=True)
    

    
    
    for s in range(1):
        #np.random.seed(s) # for the purpose of reproducibility, fix the seed
        #model._mc_fit(rating, key = label, epsilon=1e-5, maxiter=50, verbose=1)
        pred_label = model._init_task_member_ds(rating)
        task_accuracy.append(np.mean(pred_label[:, 1] == label))
        #pred_label = model._mc_infer(rating)
        #task_accuracy.append(np.mean(pred_label == label))
        
    