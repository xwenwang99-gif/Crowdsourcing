# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 13:20:57 2025

@author: wangl
"""

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
acc = np.zeros(1)
maxiter = 50
n_task, n_worker = 150,150
n_groups = 3
acc_k = []
task_accuracy = []


for i in range(3):
    
    ##rating, label = getdata1(scenario="hetero", seed=seed)
    ##getdata1
    np.random.seed(i)
    k=3

    

    alpha_0 = np.array([k, 0, 0])
    alpha_1 = np.array([0, k, 0])
    alpha_2 = np.array([0, 0, k])

    beta11 = np.array([k, 0, 0])
    beta12 = np.array([0, 0, 0])
    beta13 = np.array([0, 0, 0])

    beta21 = np.array([0, 0, 0])
    beta22 = np.array([0, k, 0])
    beta23 = np.array([0, 0, 0])

    beta31 = np.array([0, 0, 0])
    beta32 = np.array([0, 0, 0])
    beta33 = np.array([0, 0, k])

    A = np.zeros((n_task, 3))
    B1 = np.zeros((n_worker, 3))
    B2 = np.zeros((n_worker, 3))
    B3 = np.zeros((n_worker, 3))
    
    A[:n_task//3, :] = np.random.multivariate_normal(alpha_0, sigma*np.eye(3), n_task//3)
    A[n_task//3: 2*n_task//3, :] = np.random.multivariate_normal(alpha_1, sigma*np.eye(3), n_task//3)
    A[2*n_task//3:, :] = np.random.multivariate_normal(alpha_2, sigma*np.eye(3), n_task//3)

    B1[:n_worker//3, :] = np.random.multivariate_normal(beta11, sigma*np.eye(3), n_worker//3)
    B1[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta12, sigma*np.eye(3), n_worker//3)
    B1[2*n_worker//3:, :] = np.random.multivariate_normal(beta13, sigma*np.eye(3), n_worker//3)

    B2[:n_worker//3, :] = np.random.multivariate_normal(beta21, sigma*np.eye(3), n_worker//3)
    B2[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta22, sigma*np.eye(3), n_worker//3)
    B2[2*n_worker//3:, :] = np.random.multivariate_normal(beta23, sigma*np.eye(3), n_worker//3)

    B3[:n_worker//3, :] = np.random.multivariate_normal(beta31, sigma*np.eye(3), n_worker//3)
    B3[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta32, sigma*np.eye(3), n_worker//3)
    B3[2*n_worker//3:, :] = np.random.multivariate_normal(beta33, sigma*np.eye(3), n_worker//3)
    
    R_tsr = np.zeros((n_task, n_worker, 3))
    R_tsr[:, :, 0] = A.dot(B1.T)
    R_tsr[:, :, 1] = A.dot(B2.T)
    R_tsr[:, :, 2] = A.dot(B3.T)

    
    #R = np.argmax(R_tsr, axis=2)    

    exp_logits = np.exp(R_tsr)
    probs = np.zeros((n_task,n_worker,3))
    for i in range(n_task):
        for j in range(n_worker):
            probs[i,j] = exp_logits[i,j] / np.sum(exp_logits[i,j,:])
    R = np.zeros((n_task, n_worker), dtype=int)
    
    for i in range(n_task):
        for j in range(n_worker):
            R[i, j] = np.random.choice(3, p=probs[i, j, :])
    

    
    l = []

    obs_prob=1
    for i in range(n_task):

        sub_n_worker = int(n_worker * obs_prob)
        sub_worker = np.sort(np.random.choice(np.arange(n_worker), size=sub_n_worker, replace=False))
        #sub_worker = [np.repeat(1,70),np.repeat(2,70),]
        tmp = np.zeros((sub_n_worker, 3))
        tmp[:, 0] = i
        tmp[:, 1] = sub_worker
        tmp[:, 2] = R[i, sub_worker]
        l.append(tmp)

    rating = np.concatenate(l, axis=0)
    
    label = np.array([0] * (n_task // 3) + [1] * (n_task // 3) + [2] * (n_task // 3))
    
    worker_label = np.zeros((n_worker, 3))
    worker_label[:, 0] = np.array([1] * (n_worker // 3) + [0] * (n_worker // 3 * 2))
    worker_label[:, 1] = np.array([0] * (n_worker // 3) + [1] * (n_worker // 3) + ([0]*(n_worker // 3)))
    worker_label[:, 2] = np.array([0] * (n_worker // 3 * 2) + [1]*(n_worker // 3))
    
    
    #Data generation accuracy with hq workers
    MV_HQ = np.zeros(n_task)
    for t in range(n_task):
        task_t = label[t]
        hq_worker = np.where(worker_label[:, task_t] == 1)[0]        
        task_data = rating[np.isin(rating[:, 1], hq_worker) & (rating[:, 0] == t)]
        labels = task_data[:, 2]
        MV_HQ[t] = mode(labels, axis=None).mode
        
    acc_MV_HQ = np.mean(MV_HQ == label)
    
    #Data generation accuracy with all workers
    
    MV = np.zeros(n_task, dtype=int)

    for t in range(n_task):
        task_data = rating[rating[:, 0] == t]
        labels = task_data[:, 2]
        MV[t] = mode(labels, axis=None).mode
        
    acc_MV = np.mean(MV == label)


    model = LFGP(lf_dim=3, n_worker_group=3, lambda1=1, lambda2=1)
    model._prescreen(rating)
    
    _, task_id = np.unique(rating[:, 0], return_inverse=True)
    _, worker_id = np.unique(rating[:, 1], return_inverse=True)
    
    trueloss = 0
    
    for i in range(len(rating[:, 0])):    
        prob = [A[task_id[i], :].dot(B1[worker_id[i], :]),A[task_id[i], :].dot(B2[worker_id[i], :]),A[task_id[i], :].dot(B3[worker_id[i], :])]
        prob = np.divide(np.exp(prob), np.sum(np.exp(prob)))
    
        trueloss += - np.log(prob[int(rating[i, 2])])
    
    
    
    
    for s in range(1):
        #np.random.seed(s) # for the purpose of reproducibility, fix the seed
        model._mc_fit(rating, key = label, epsilon=1e-5, maxiter=50, verbose=1)
        #pred_label = model._init_task_member_ds(rating)
        #task_accuracy.append(np.mean(pred_label[:, 1] == label))
        pred_label = model._mc_infer(rating)
        task_accuracy.append(np.mean(pred_label == label))
        
    