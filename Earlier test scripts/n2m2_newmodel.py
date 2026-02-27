# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:19:04 2024

@author: wangl
"""

from src.lfgp_withoutO import LFGP
import numpy as np
import warnings
import numpy_indexed as npi
import matplotlib.pyplot as plt

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
        tmp = np.zeros((sub_n_worker, 3))
        tmp[:, 0] = i
        tmp[:, 1] = sub_worker
        tmp[:, 2] = R[i, sub_worker]
        l.append(tmp)

    rating = np.concatenate(l, axis=0)
    
    label = np.array([0] * (n_task // 2) + [1] * (n_task // 2))
    #worker_label= np.array([0] * (n_worker // 5) + [1] * (n_worker // 5) + [2] * (n_worker // 5)+ [3] * (n_worker// 5)+ [4] * (n_worker // 5))
    worker_label = np.zeros((n_worker, 5))
    worker_label[:, 0] = np.array([1] * (n_worker // 3) + [0]*(n_worker // 3 * 2))
    worker_label[:, 1] = np.array([0] * (n_worker // 3) + [1]*(n_worker // 3)+ [0]*(n_worker // 3))
    
    
    
    
    
    ######################getdata1 over
    
    ######LFGP

    model = LFGP(lf_dim=2, n_worker_group=2, lambda1=1, lambda2=1)
    model._prescreen(rating)

    
    
    tmp = []
    new_tmp = []
    old_tmp = []
    l=[]
    U_record = np.zeros((n_task, 5))
    V_record = np.zeros((n_worker, 2, 5))
    worker_acc_record = []
    for s in range(1):        
        np.random.seed(s) # for the purpose of reproducibility, fix the seed
        A, B, U, V= model._mc_fit(rating, key = label, epsilon=1e-4, maxiter=50, verbose=1)
        #U_record[:, s] = U
        #V_record[:, :, s] = V
        U_new = model.label_swap(U, label)
        worker_acc = model.calculate_worker_accuracy(worker_label)
        worker_acc_record.append(worker_acc)
        U_mv = model._mc_infer(rating)
        tmp.append(np.mean(U_new == label))
        new_tmp.append(np.mean(U_mv == label))
        old_tmp.append(np.mean(U == label))
        #worker_acc = model.accuracy_worker(rating, key = label)
        #A0 = A[U == 0]
        #A1 = A[U == 1]
        #plt.scatter(A0[:, 0],A0[:, 1])
        #plt.scatter(A1[:, 0],A1[:, 1])
        #plt.show()