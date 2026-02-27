# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 13:03:40 2026

@author: wangl
"""


import numpy as np





def getdata(
    n_task: int,
    n_worker: int,
    n_task_groups: int,
    n_worker_groups: int,    
    k: float = 3.0,
    sigma: float = 1.0,
    obs_prob=1,
    noise_group: int = 0
):
    alpha_0 = np.array([k,0,0,0,0])
    alpha_1 = np.array([0,k,0,0,0])
    alpha_2 = np.array([0,0,k,0,0])
    alpha_3 = np.array([0,0,0,k,0])
    alpha_4 = np.array([0,0,0,0,k])
    
    if noise_group==0:

        beta11 = np.array([k,0,0,0,0])
        beta12 = np.array([0,0,0,0,0])
        beta13 = np.array([0,0,0,0,0])
        beta14 = np.array([0,0,0,0,0])
        beta15 = np.array([0,0,0,0,0])
    
    
        beta21 = np.array([0,0,0,0,0])
        beta22 = np.array([0,k,0,0,0])
        beta23 = np.array([0,0,0,0,0])
        beta24 = np.array([0,0,0,0,0])
        beta25 = np.array([0,0,0,0,0])
    
    
        beta31 = np.array([0,0,0,0,0])
        beta32 = np.array([0,0,0,0,0])
        beta33 = np.array([0,0,k,0,0])
        beta34 = np.array([0,0,0,0,0])
        beta35 = np.array([0,0,0,0,0])
    
        
        beta41 = np.array([0,0,0,0,0])
        beta42 = np.array([0,0,0,0,0])
        beta43 = np.array([0,0,0,0,0])
        beta44 = np.array([0,0,0,k,0])
        beta45 = np.array([0,0,0,0,0])
    
        
        beta51 = np.array([0,0,0,0,0])
        beta52 = np.array([0,0,0,0,0])
        beta53 = np.array([0,0,0,0,0])
        beta54 = np.array([0,0,0,0,0])
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
    
        B2[:n_worker//5, :] = np.random.random((n_worker//5, 5)) * 2 * k - k
        B2[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta22, sigma *np.eye(5), n_worker//5)
        B2[2*n_worker//5: , :] = np.random.random((n_worker*3//5, 5)) * 2 * k - k
        
        B3[:2*n_worker//5, :] = np.random.random((2*n_worker//5, 5)) * 2 * k - k
        B3[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta33,sigma * np.eye(5), n_worker//5)
        B3[3*n_worker//5: , :] = np.random.random((2*n_worker//5, 5)) * 2 * k - k
        
        B4[:3*n_worker//5, :] = np.random.random((n_worker*3//5, 5)) * 2 * k - k
        B4[3*n_worker//5:4*n_worker//5, :] = np.random.multivariate_normal(beta44, sigma *np.eye(5), n_worker//5)
        B4[4*n_worker//5: , :] = np.random.random((n_worker//5, 5)) * 2 * k - k
        
        B5[:4*n_worker//5, :] = np.random.random((4*n_worker//5, 5)) * 2 * k - k
        B5[4*n_worker//5: , :] = np.random.multivariate_normal(beta55, sigma *np.eye(5), n_worker//5)
        
                
        worker_label = np.zeros((n_worker, n_task_groups))
        worker_label[:, 0] = np.array([1] * (n_worker // 5) + [0] * (n_worker // 5 * 4))
        worker_label[:, 1] = np.array([0] * (n_worker // 5) + [1] * (n_worker // 5) + ([0]*(n_worker // 5*3)))
        worker_label[:, 2] = np.array([0] * (n_worker // 5 * 2) + [1]*(n_worker // 5) + [0]*(n_worker // 5 * 2))
        worker_label[:, 3] = np.array([0] * (n_worker // 5 * 3) + [1]*(n_worker // 5) + [0]*(n_worker // 5 ))
        worker_label[:, 4] = np.array([0] * (n_worker // 5 * 4) + [1] * (n_worker // 5))
    if noise_group==2:
        beta11 = np.array([k,0,0,0,0])
        beta12 = np.array([0,0,0,0,0])
        beta13 = np.array([0,0,0,0,0])
        beta14 = np.array([0,0,0,0,0])
        beta15 = np.array([0,0,0,0,0])
        beta16 = np.array([0,0,0,0,0])
        beta17 = np.array([0,0,0,0,0])
    
    
        beta21 = np.array([0,0,0,0,0])
        beta22 = np.array([0,k,0,0,0])
        beta23 = np.array([0,0,0,0,0])
        beta24 = np.array([0,0,0,0,0])
        beta25 = np.array([0,0,0,0,0])
        beta26 = np.array([0,0,0,0,0])
        beta27 = np.array([0,0,0,0,0])
    
        beta31 = np.array([0,0,0,0,0])
        beta32 = np.array([0,0,0,0,0])
        beta33 = np.array([0,0,k,0,0])
        beta34 = np.array([0,0,0,0,0])
        beta35 = np.array([0,0,0,0,0])
        beta36 = np.array([0,0,0,0,0])
        beta37 = np.array([0,0,0,0,0])
    
        
        beta41 = np.array([0,0,0,0,0])
        beta42 = np.array([0,0,0,0,0])
        beta43 = np.array([0,0,0,0,0])
        beta44 = np.array([0,0,0,k,0])
        beta45 = np.array([0,0,0,0,0])
        beta46 = np.array([0,0,0,0,0])
        beta47 = np.array([0,0,0,0,0])
    
        
        beta51 = np.array([0,0,0,0,0])
        beta52 = np.array([0,0,0,0,0])
        beta53 = np.array([0,0,0,0,0])
        beta54 = np.array([0,0,0,0,0])
        beta55 = np.array([0,0,0,0,k])
        beta56 = np.array([0,0,0,0,0])
        beta57 = np.array([0,0,0,0,0])
    
    
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
    
        B1[:n_worker//7, :] = np.random.multivariate_normal(beta11, sigma *np.eye(5), n_worker//7)
        B1[n_worker//7: , :] = np.random.random((n_worker*6//7, 5)) * 2 * k - k
    
        B2[:n_worker//7, :] = np.random.random((n_worker//7, 5)) * 2 * k - k
        B2[n_worker//7: 2*n_worker//7, :] = np.random.multivariate_normal(beta22, sigma *np.eye(5), n_worker//7)
        B2[2*n_worker//7: , :] = np.random.random((n_worker*5//7, 5)) * 2 * k - k
        
        B3[:2*n_worker//7, :] = np.random.random((2*n_worker//7, 5)) * 2 * k - k
        B3[2*n_worker//7: 3*n_worker//7, :] = np.random.multivariate_normal(beta33,sigma * np.eye(5), n_worker//7)
        B3[3*n_worker//7: , :] = np.random.random((4*n_worker//7, 5)) * 2 * k - k
        
        B4[:3*n_worker//7, :] = np.random.random((n_worker*3//7, 5)) * 2 * k - k
        B4[3*n_worker//7:4*n_worker//7, :] = np.random.multivariate_normal(beta44, sigma *np.eye(5), n_worker//7)
        B4[4*n_worker//7: , :] = np.random.random((3*n_worker//7, 5)) * 2 * k - k
        
        B5[:4*n_worker//7, :] = np.random.random((4*n_worker//7, 5)) * 2 * k - k
        B5[4*n_worker//7:5*n_worker//7 , :] = np.random.multivariate_normal(beta55, sigma *np.eye(5), n_worker//7)
        B5[5*n_worker//7: , :] = np.random.random((2*n_worker//7, 5)) * 2 * k - k  
   
        worker_label = np.zeros((n_worker, n_task_groups))
        worker_label[:, 0] = np.array([1] * (n_worker // 7) + [0] * (n_worker // 7 * 6))
        worker_label[:, 1] = np.array([0] * (n_worker // 7) + [1] * (n_worker // 7) + ([0]*(n_worker // 7*5)))
        worker_label[:, 2] = np.array([0] * (n_worker // 7 * 2) + [1]*(n_worker // 7) + [0]*(n_worker // 7 * 4))
        worker_label[:, 3] = np.array([0] * (n_worker // 7 * 3) + [1]*(n_worker // 7) + [0]*(n_worker // 7 * 3))
        worker_label[:, 4] = np.array([0] * (n_worker // 7 * 4) + [1] * (n_worker // 7) + [0]*(n_worker // 7 * 2))
    
    if noise_group==5:
        beta11 = np.array([k,0,0,0,0])
        beta12 = np.array([0,0,0,0,0])
        beta13 = np.array([0,0,0,0,0])
        beta14 = np.array([0,0,0,0,0])
        beta15 = np.array([0,0,0,0,0])
        beta16 = np.array([0,0,0,0,0])
        beta17 = np.array([0,0,0,0,0])
        beta18 = np.array([0,0,0,0,0])
        beta19 = np.array([0,0,0,0,0])
        beta10 = np.array([0,0,0,0,0])
    
    
        beta21 = np.array([0,0,0,0,0])
        beta22 = np.array([0,k,0,0,0])
        beta23 = np.array([0,0,0,0,0])
        beta24 = np.array([0,0,0,0,0])
        beta25 = np.array([0,0,0,0,0])
        beta26 = np.array([0,0,0,0,0])
        beta27 = np.array([0,0,0,0,0])
        beta28 = np.array([0,0,0,0,0])
        beta29 = np.array([0,0,0,0,0])
        beta20 = np.array([0,0,0,0,0])
    
        beta31 = np.array([0,0,0,0,0])
        beta32 = np.array([0,0,0,0,0])
        beta33 = np.array([0,0,k,0,0])
        beta34 = np.array([0,0,0,0,0])
        beta35 = np.array([0,0,0,0,0])
        beta36 = np.array([0,0,0,0,0])
        beta37 = np.array([0,0,0,0,0])
        beta38 = np.array([0,0,0,0,0])
        beta39 = np.array([0,0,0,0,0])
        beta30 = np.array([0,0,0,0,0])
    
        
        beta41 = np.array([0,0,0,0,0])
        beta42 = np.array([0,0,0,0,0])
        beta43 = np.array([0,0,0,0,0])
        beta44 = np.array([0,0,0,k,0])
        beta45 = np.array([0,0,0,0,0])
        beta46 = np.array([0,0,0,0,0])
        beta47 = np.array([0,0,0,0,0])
        beta48 = np.array([0,0,0,0,0])
        beta49 = np.array([0,0,0,0,0])
        beta40 = np.array([0,0,0,0,0])
    
        
        beta51 = np.array([0,0,0,0,0])
        beta52 = np.array([0,0,0,0,0])
        beta53 = np.array([0,0,0,0,0])
        beta54 = np.array([0,0,0,0,0])
        beta55 = np.array([0,0,0,0,k])
        beta56 = np.array([0,0,0,0,0])
        beta57 = np.array([0,0,0,0,0])
        beta58 = np.array([0,0,0,0,0])
        beta59 = np.array([0,0,0,0,0])
        beta50 = np.array([0,0,0,0,0])
    
    
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
    
        B1[:n_worker//10, :] = np.random.multivariate_normal(beta11, sigma *np.eye(5), n_worker//10)
        B1[n_worker//10: , :] = np.random.random((n_worker*9//10, 5)) * 2 * k - k
    
        B2[:n_worker//10, :] = np.random.random((n_worker//10, 5)) * 2 * k - k
        B2[n_worker//10: 2*n_worker//10, :] = np.random.multivariate_normal(beta22, sigma *np.eye(5), n_worker//10)
        B2[2*n_worker//10: , :] = np.random.random((n_worker*8//10, 5)) * 2 * k - k
        
        B3[:2*n_worker//10, :] = np.random.random((2*n_worker//10, 5)) * 2 * k - k
        B3[2*n_worker//10: 3*n_worker//10, :] = np.random.multivariate_normal(beta33,sigma * np.eye(5), n_worker//10)
        B3[3*n_worker//10: , :] = np.random.random((7*n_worker//10, 5)) * 2 * k - k
        
        B4[:3*n_worker//10, :] = np.random.random((n_worker*3//10, 5)) * 2 * k - k
        B4[3*n_worker//10:4*n_worker//10, :] = np.random.multivariate_normal(beta44, sigma *np.eye(5), n_worker//10)
        B4[4*n_worker//10: , :] = np.random.random((6*n_worker//10, 5)) * 2 * k - k
        
        B5[:4*n_worker//10, :] = np.random.random((4*n_worker//10, 5)) * 2 * k - k
        B5[4*n_worker//10:5*n_worker//10 , :] = np.random.multivariate_normal(beta55, sigma *np.eye(5), n_worker//10)
        B5[5*n_worker//10: , :] = np.random.random((5*n_worker//10, 5)) * 2 * k - k  
   
        worker_label = np.zeros((n_worker, n_task_groups))
        worker_label[:, 0] = np.array([1] * (n_worker // 10) + [0] * (n_worker // 10 * 9))
        worker_label[:, 1] = np.array([0] * (n_worker // 10) + [1] * (n_worker // 10) + ([0]*(n_worker // 10*8)))
        worker_label[:, 2] = np.array([0] * (n_worker // 10 * 2) + [1]*(n_worker // 10) + [0]*(n_worker // 10 * 7))
        worker_label[:, 3] = np.array([0] * (n_worker // 10 * 3) + [1]*(n_worker // 10) + [0]*(n_worker // 10 * 6))
        worker_label[:, 4] = np.array([0] * (n_worker // 10 * 4) + [1] * (n_worker // 10) + [0]*(n_worker // 10 * 5))
    
    label = np.array([0] * (n_task // 5) + [1] * (n_task // 5) + [2] * (n_task // 5)+ [3] * (n_task // 5)+ [4] * (n_task // 5))
    
    R_tsr = np.zeros((n_task, n_worker, 5))
    R_tsr[:, :, 0] = A.dot(B1.T)
    R_tsr[:, :, 1] = A.dot(B2.T)
    R_tsr[:, :, 2] = A.dot(B3.T)
    R_tsr[:, :, 3] = A.dot(B4.T)
    R_tsr[:, :, 4] = A.dot(B5.T)
    
    exp_logits = np.exp(R_tsr)
    probs = np.zeros((n_task,n_worker,5))
    for i in range(n_task):
        for j in range(n_worker):
            probs[i,j] = exp_logits[i,j] / np.sum(exp_logits[i,j,:])
    R = np.zeros((n_task, n_worker), dtype=int)
    
    for i in range(n_task):
        for j in range(n_worker):
            R[i, j] = np.random.choice(5, p=probs[i, j, :])

    l = []

    
    for i in range(n_task):

        sub_n_worker = int(n_worker * obs_prob)
        sub_worker = np.sort(np.random.choice(np.arange(n_worker), size=sub_n_worker, replace=False))
        #sub_worker = [np.repeat(1,70),np.repeat(2,70),]
        tmp_data = np.zeros((sub_n_worker, 3))
        tmp_data[:, 0] = i
        tmp_data[:, 1] = sub_worker
        tmp_data[:, 2] = R[i, sub_worker]
        l.append(tmp_data)

    rating = np.concatenate(l, axis=0)
    
    R_obs = np.full((n_task, n_worker), np.nan)

    task_ids = rating[:, 0].astype(int)
    worker_ids = rating[:, 1].astype(int)
    labels_obs = rating[:, 2].astype(int)
    
    R_obs[task_ids, worker_ids] = labels_obs
    
    
    
    return rating, label, worker_label, R_obs
    
