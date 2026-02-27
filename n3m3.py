# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:23:52 2024

@author: wangl
"""

from src.lfgp_paper_honest_infer import LFGP
import numpy as np
import warnings
import numpy_indexed as npi

import numpy as np

warnings.filterwarnings('ignore')

acc = np.zeros(1)

for seed in range(1):
    
    ##rating, label = getdata1(scenario="hetero", seed=seed)
    ##getdata1
    if seed is not None:
        np.random.seed(seed)

    n_task, n_worker = 150, 150

    alpha_0 = np.array([2, 0, 0])
    alpha_1 = np.array([0, 2, 0])
    alpha_2 = np.array([0, 0, 2])

    beta11 = np.array([2, 0, 0])
    beta12 = np.array([0, 1, 1])
    beta13 = np.array([1, 1, 0])

    beta21 = np.array([1, 1, 1])
    beta22 = np.array([0, 2, 0])
    beta23 = np.array([1, 1, 0])

    beta31 = np.array([1, 1, 0])
    beta32 = np.array([1, 1, 0])
    beta33 = np.array([0, 0, 2])

    A = np.zeros((n_task, 3))
    B1 = np.zeros((n_worker, 3))
    B2 = np.zeros((n_worker, 3))
    B3 = np.zeros((n_worker, 3))
    
    A[:n_task//3, :] = np.random.multivariate_normal(alpha_0, 2 * np.eye(3), n_task//3)
    A[n_task//3: 2*n_task//3, :] = np.random.multivariate_normal(alpha_1, 2 * np.eye(3), n_task//3)
    A[2*n_task//3:, :] = np.random.multivariate_normal(alpha_2, 2 * np.eye(3), n_task//3)

    B1[:n_worker//3, :] = np.random.multivariate_normal(beta11, np.eye(3), n_worker//3)
    B1[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta12, np.eye(3), n_worker//3)
    B1[2*n_worker//3:, :] = np.random.multivariate_normal(beta13, np.eye(3), n_worker//3)

    B2[:n_worker//3, :] = np.random.multivariate_normal(beta21, np.eye(3), n_worker//3)
    B2[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta22, np.eye(3), n_worker//3)
    B2[2*n_worker//3:, :] = np.random.multivariate_normal(beta23, np.eye(3), n_worker//3)

    B3[:n_worker//3, :] = np.random.multivariate_normal(beta31, np.eye(3), n_worker//3)
    B3[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta32, np.eye(3), n_worker//3)
    B3[2*n_worker//3:, :] = np.random.multivariate_normal(beta33, np.eye(3), n_worker//3)

    R_tsr = np.zeros((n_task, n_worker, 3))
    R_tsr[:, :, 0] = A.dot(B1.T)
    R_tsr[:, :, 1] = A.dot(B2.T)
    R_tsr[:, :, 2] = A.dot(B3.T)
    R = np.argmax(R_tsr, axis=2)

    l = []
    obs_prob=0.3
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
    ######################getdata1 over
    
    ######LFGP

    model = LFGP(lf_dim=3, n_worker_group=3, lambda1=1, lambda2=1)
    model._prescreen(rating)
    
    _, task_id = np.unique(rating[:, 0], return_inverse=True)
    _, worker_id = np.unique(rating[:, 1], return_inverse=True)

    trueloss = 0

    for i in range(len(rating[:, 0])):    
        prob = [A[task_id[i], :].dot(B1[worker_id[i], :]),A[task_id[i], :].dot(B2[worker_id[i], :]),A[task_id[i], :].dot(B3[worker_id[i], :])]
        prob = np.divide(np.exp(prob), np.sum(np.exp(prob)))

        trueloss += - np.log(prob[int(rating[i, 2])])
    
    
    tmp = []
    l=[]
    for s in range(1):
        np.random.seed(s) # for the purpose of reproducibility, fix the seed
        model._mc_fit(rating, label, epsilon=1e-4, maxiter=50, verbose=1)
        
        candidate, assignment, a, b = model._mc_infer(rating, label)
        #tmp.append(np.mean(pred_label[:, 1] == label))
        _, alpha = npi.group_by(label).mean(A, axis=0)
        _, beta = npi.group_by(label).mean(B1, axis=0)
    #acc[seed] = np.max(tmp)
    
    
#print("---- accuracy: {0}({1}) ----".format(np.mean(acc), np.std(acc)))