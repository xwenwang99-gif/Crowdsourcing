# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:53:14 2025

@author: wangl
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:19:04 2024

@author: wangl
"""

from src.lfgp_paper import LFGP
import numpy as np
import warnings

import numpy as np

warnings.filterwarnings('ignore')
k=1
acc = np.zeros(1)

for seed in range(1):
    
    ##rating, label = getdata1(scenario="hetero", seed=seed)
    ##getdata1
    if seed is not None:
        np.random.seed(seed)

    n_task, n_worker = 200, 120

    alpha_0 = np.array([k,0])
    alpha_1 = np.array([0,k])

    beta11 = np.array([k,0])
    beta12 = np.array([0,0])
    beta13 = np.array([0,0])


    beta21 = np.array([0,0])
    beta22 = np.array([0,k])
    beta23 = np.array([0,0])

    A = np.zeros((n_task, 2))
    B1 = np.zeros((n_worker, 2))
    B2 = np.zeros((n_worker, 2))

    A[:n_task//2, :] = np.random.multivariate_normal(alpha_0, 0.1 * np.eye(2), n_task//2)
    A[n_task//2:, :] = np.random.multivariate_normal(alpha_1, 0.1 * np.eye(2), n_task//2)
    
    B1[:n_worker//3, :] = np.random.multivariate_normal(beta11, 0.1 *np.eye(2), n_worker//3)
    B1[n_worker//3: , :] = np.random.random((n_worker//3*2, 2)) * 2 * k - k

    B2[:n_worker//3, :] = np.random.random((n_worker//3, 2)) * 2 * k - k
    B2[n_worker//3: 2*n_worker//3, :] = np.random.multivariate_normal(beta22, 0.1 *np.eye(2), n_worker//3)
    B2[2*n_worker//3: , :] = np.random.random((n_worker//3, 2)) * 2 * k - k
    
    R_tsr = np.zeros((n_task, n_worker, 2))
    R_tsr[:, :, 0] = A.dot(B1.T)
    R_tsr[:, :, 1] = A.dot(B2.T)
    R = np.argmax(R_tsr, axis=2)

    l = []
    obs_prob=0.5
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
    
    label = np.array([0] * (n_task // 2) + [1] * (n_task // 2))
    worker_label = np.array([0] * (n_worker // 3) + [1] * (n_worker // 3)+ [2] * (n_worker // 3))
    ######################getdata1 over
    
    ######LFGP

    model = LFGP(lf_dim=2, n_worker_group=3, lambda1=1, lambda2=1)
    model._prescreen(rating)
    
    _, task_id = np.unique(rating[:, 0], return_inverse=True)
    _, worker_id = np.unique(rating[:, 1], return_inverse=True)

    trueloss = 0

    for i in range(len(rating[:, 0])):    
        prob = [A[task_id[i], :].dot(B1[worker_id[i], :]),A[task_id[i], :].dot(B2[worker_id[i], :])]
        prob = np.divide(np.exp(prob), np.sum(np.exp(prob)))

        trueloss += - np.log(prob[int(rating[i, 2])])
    
    
    tmp_task = []
    tmp_worker = []
    l=[]
    U_record = np.zeros((n_task, 3))
    V_record = np.zeros((n_worker, 3))
    for s in range(3):
        np.random.seed(s) # for the purpose of reproducibility, fix the seed
        model._mc_fit(rating, key = label, epsilon=1e-4, maxiter=50, verbose=1)
        
        U_record[:, s], V_record[:, s], pred_label = model._mc_infer(rating, worker_label)
        
        tmp_worker.append(np.mean(V_record[:, s] == worker_label))
        tmp_task.append(np.mean(pred_label[:, 1] == label))
