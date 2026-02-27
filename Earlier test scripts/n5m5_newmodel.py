# %%
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
import pandas as pd

import numpy as np

warnings.filterwarnings('ignore')
k=3
sigma = 0.5
acc = np.zeros(1)

for seed in range(1):
    
    ##rating, label = getdata1(scenario="hetero", seed=seed)
    ##getdata1
    if seed is not None:
        np.random.seed(seed)

    n_task, n_worker = 200, 200

    alpha_0 = np.array([k,0,0,0,0])
    alpha_1 = np.array([0,k,0,0,0])
    alpha_2 = np.array([0,0,k,0,0])
    alpha_3 = np.array([0,0,0,k,0])
    alpha_4 = np.array([0,0,0,0,k])

    beta11 = np.array([k,0,0,0,0])
    beta12 = np.array([0,0,0,0,0])
    beta13 = np.array([0,0,0,0,0])
    beta14 = np.array([0,0,0,0,0])
    beta15 = np.array([0,0,0,0,0])


    beta21 = np.array([0,k,0,0,0])
    beta22 = np.array([0,k,0,0,0])
    beta23 = np.array([0,0,0,0,0])
    beta24 = np.array([0,0,0,0,0])
    beta25 = np.array([0,0,0,0,0])


    beta31 = np.array([0,0,0,0,0])
    beta32 = np.array([0,0,k,0,0])
    beta33 = np.array([0,0,k,0,0])
    beta34 = np.array([0,0,k,0,0])
    beta35 = np.array([0,0,0,0,0])

    
    beta41 = np.array([0,0,0,0,0])
    beta42 = np.array([0,0,0,0,0])
    beta43 = np.array([0,0,0,0,0])
    beta44 = np.array([0,0,0,0,0])
    beta45 = np.array([0,0,0,k,0])

    
    beta51 = np.array([0,0,0,0,k])
    beta52 = np.array([0,0,0,0,0])
    beta53 = np.array([0,0,0,0,0])
    beta54 = np.array([0,0,0,0,k])
    beta55 = np.array([0,0,0,0,k])


    A = np.zeros((n_task, 5))
    B1 = np.zeros((n_worker, 5))
    B2 = np.zeros((n_worker, 5))
    B3 = np.zeros((n_worker, 5))
    B4 = np.zeros((n_worker, 5))
    B5 = np.zeros((n_worker, 5))

    A[:n_task//5, :] = np.random.multivariate_normal(alpha_0, sigma * np.eye(5), n_task//5)
    A[n_task//5: 2*n_task//5, :] = np.random.multivariate_normal(alpha_1, sigma * np.eye(5), n_task//5)
    A[2*n_task//5: 3*n_task//5, :] = np.random.multivariate_normal(alpha_2, sigma *np.eye(5), n_task//5)
    A[3*n_task//5: 4*n_task//5, :] = np.random.multivariate_normal(alpha_3, sigma * np.eye(5), n_task//5)
    A[4*n_task//5:, :] = np.random.multivariate_normal(alpha_4, sigma * np.eye(5), n_task//5)

    B1[:n_worker//5, :] = np.random.multivariate_normal(beta11, sigma *np.eye(5), n_worker//5)
    B1[n_worker//5: , :] = np.random.random((n_worker*4//5, 5)) * 2 * k - k

    B2[:n_worker//5, :] = np.random.multivariate_normal(beta21, sigma *np.eye(5), n_worker//5)
    B2[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta22, sigma *np.eye(5), n_worker//5)
    B2[2*n_worker//5: , :] = np.random.random((n_worker*3//5, 5)) * 2 * k - k
    
    B3[:n_worker//5, :] = np.random.random((n_worker//5, 5)) * 2 * k - k
    B3[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta32,sigma * np.eye(5), n_worker//5)
    B3[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta33, sigma *np.eye(5), n_worker//5)
    B3[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta34, sigma *np.eye(5), n_worker//5)
    B3[4*n_worker//5: , :] = np.random.random((n_worker//5, 5)) * 2 * k - k
    
    B4[:4*n_worker//5, :] = np.random.random((n_worker*4//5, 5)) * 2 * k - k
    B4[4*n_worker//5: , :] = np.random.multivariate_normal(beta45, sigma *np.eye(5), n_worker//5)
    
    B5[:n_worker//5, :] = np.random.multivariate_normal(beta51, sigma *np.eye(5), n_worker//5)
    B5[n_worker//5: 3*n_worker//5, :] = np.random.random((n_worker*2//5, 5)) * 2 * k - k
    B5[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta54, sigma *np.eye(5), n_worker//5)
    B5[4*n_worker//5: , :] = np.random.multivariate_normal(beta55, sigma *np.eye(5), n_worker//5)
    
    R_tsr = np.zeros((n_task, n_worker, 5))
    R_tsr[:, :, 0] = A.dot(B1.T)
    R_tsr[:, :, 1] = A.dot(B2.T)
    R_tsr[:, :, 2] = A.dot(B3.T)
    R_tsr[:, :, 3] = A.dot(B4.T)
    R_tsr[:, :, 4] = A.dot(B5.T)
    R = np.argmax(R_tsr, axis=2)

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
    
    label = np.array([0] * (n_task // 5) + [1] * (n_task // 5) + [2] * (n_task // 5)+ [3] * (n_task // 5)+ [4] * (n_task // 5))
    #worker_label= np.array([0] * (n_worker // 5) + [1] * (n_worker // 5) + [2] * (n_worker // 5)+ [3] * (n_worker// 5)+ [4] * (n_worker // 5))
    worker_label = np.zeros((n_worker, 5))
    worker_label[:, 0] = np.array([1] * (n_worker // 5) + [0]*(n_worker // 5 * 4))
    worker_label[:, 1] = np.array([1] * (n_worker // 5*2) + ([0]*(n_worker // 5*3)))
    worker_label[:, 2] = np.array([0] * (n_worker // 5) + [1]*(n_worker // 5*3) + [0]*(n_worker // 5))
    worker_label[:, 3] = np.array([0] * (n_worker // 5*4) + [1]*(n_worker // 5))
    worker_label[:, 4] = np.array([1] * (n_worker // 5) + [0] * (n_worker // 5*2) + [1] * (n_worker // 5*2))
        
    
    
    
    
    ######################getdata1 over
    
    ######LFGP

    model = LFGP(lf_dim=5, n_worker_group=5, lambda1=1, lambda2=1)
    model._prescreen(rating)
    
    _, task_id = np.unique(rating[:, 0], return_inverse=True)
    _, worker_id = np.unique(rating[:, 1], return_inverse=True)

    trueloss = 0

    for i in range(len(rating[:, 0])):    
        prob = [A[task_id[i], :].dot(B1[worker_id[i], :]),A[task_id[i], :].dot(B2[worker_id[i], :]),A[task_id[i], :].dot(B3[worker_id[i], :]),A[task_id[i], :].dot(B4[worker_id[i], :]),A[task_id[i], :].dot(B5[worker_id[i], :])]
        prob = np.divide(np.exp(prob), np.sum(np.exp(prob)))

        trueloss += - np.log(prob[int(rating[i, 2])])
    
    
    tmp = []
    new_tmp = []
    old_tmp = []
    l=[]
    U_record = np.zeros((n_task, 5))
    V_record = np.zeros((n_worker, 5, 1))
    B0 = np.zeros((n_worker, 2))
    B1 = np.zeros((n_worker, 2))
    worker_acc_record = []
    for s in range(1):        
        np.random.seed(4) # for the purpose of reproducibility, fix the seed
        A, B, U, V, centers= model._mc_fit(rating, key = label, epsilon=1e-4, maxiter=50, verbose=0)
        U_record[:, s] = U
        V_record[:, :, s] = V
        U_new = model.label_swap(U, label)
        worker_acc = model.calculate_worker_accuracy(worker_label)
        #df = pd.DataFrame(worker_acc)
        #latex_code = df.to_latex(index=False, header=False)  # Remove index and header for cleaner output
        #print(latex_code)
        #worker_acc_record.append(worker_acc)
        U_mv = model._mc_infer(rating)
        tmp.append(np.mean(U_new == label))
        new_tmp.append(np.mean(U_mv == label))
        old_tmp.append(np.mean(U == label))

        model.diagnosis(worker_label, rating)
        model.distance_graph(centers, worker_label)
        #worker_acc = model.accuracy_worker(rating, key = label)