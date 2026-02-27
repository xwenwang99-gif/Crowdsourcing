from src.lfgp_paper_honest_infer import LFGP
import numpy as np
import warnings

import numpy as np

warnings.filterwarnings('ignore')
k=3
acc = np.zeros(1)

for seed in range(1):
    
    ##rating, label = getdata1(scenario="hetero", seed=seed)
    ##getdata1
    if seed is not None:
        np.random.seed(seed)

    n_task, n_worker = 200, 500

    alpha_0 = np.array([k,0,0,k,0])
    alpha_1 = np.array([k,k,0,0,0])
    alpha_2 = np.array([0,k,k,0,0])
    alpha_3 = np.array([0,0,0,k,k])
    alpha_4 = np.array([k,0,0,0,k])

    beta11 = np.array([k,0,0,k,0])
    beta12 = np.array([0,0,k,0,0])
    beta13 = np.array([0,k,0,0,0])
    beta14 = np.array([0,0,k,0,0])
    beta15 = np.array([0,0,0,0,k])


    beta21 = np.array([k,k,0,0,0])
    beta22 = np.array([k,k,0,0,0])
    beta23 = np.array([0,0,k,0,0])
    beta24 = np.array([0,0,k,0,k])
    beta25 = np.array([0,0,0,k,0])


    beta31 = np.array([k,0,0,0,0])
    beta32 = np.array([0,k,k,0,0])
    beta33 = np.array([0,k,k,0,0])
    beta34 = np.array([0,k,k,0,0])
    beta35 = np.array([k,0,0,0,0])

    
    beta41 = np.array([0,k,0,0,0])
    beta42 = np.array([k,0,k,0,0])
    beta43 = np.array([0,k,0,0,0])
    beta44 = np.array([k,0,0,0,0])
    beta45 = np.array([0,0,0,k,k])

    
    beta51 = np.array([k,0,0,0,k])
    beta52 = np.array([0,k,0,0,0])
    beta53 = np.array([0,0,k,0,0])
    beta54 = np.array([k,0,0,0,k])
    beta55 = np.array([k,0,0,0,k])


    A = np.zeros((n_task, 5))
    B1 = np.zeros((n_worker, 5))
    B2 = np.zeros((n_worker, 5))
    B3 = np.zeros((n_worker, 5))
    B4 = np.zeros((n_worker, 5))
    B5 = np.zeros((n_worker, 5))
    '''
    A[:n_task//5, :] = np.random.multivariate_normal(alpha_0, 0.5 * np.eye(5), n_task//5)
    A[n_task//5: 2*n_task//5, :] = np.random.multivariate_normal(alpha_1, 0.5 * np.eye(5), n_task//5)
    A[2*n_task//5: 3*n_task//5, :] = np.random.multivariate_normal(alpha_2, 0.5 *np.eye(5), n_task//5)
    A[3*n_task//5: 4*n_task//5, :] = np.random.multivariate_normal(alpha_3, 0.5 * np.eye(5), n_task//5)
    A[4*n_task//5:, :] = np.random.multivariate_normal(alpha_4, 0.5 * np.eye(5), n_task//5)

    B1[:n_worker//5, :] = np.random.multivariate_normal(beta11, 0.5 *np.eye(5), n_worker//5)
    B1[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta12, 0.5 *np.eye(5), n_worker//5)
    B1[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta13, 0.5 *np.eye(5), n_worker//5)
    B1[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta14,0.5 *np.eye(5), n_worker//5)
    B1[4*n_worker//5: , :] = np.random.multivariate_normal(beta15, 0.5 *np.eye(5), n_worker//5)
    
    B2[:n_worker//5, :] = np.random.multivariate_normal(beta21, 0.5 *np.eye(5), n_worker//5)
    B2[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta22, 0.5 *np.eye(5), n_worker//5)
    B2[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta23, 0.5 *np.eye(5), n_worker//5)
    B2[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta24, 0.5 *np.eye(5), n_worker//5)
    B2[4*n_worker//5: , :] = np.random.multivariate_normal(beta25, 0.5 *np.eye(5), n_worker//5)
    
    B3[:n_worker//5, :] = np.random.multivariate_normal(beta31, 0.5 *np.eye(5), n_worker//5)
    B3[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta32,0.5 * np.eye(5), n_worker//5)
    B3[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta33, 0.5 *np.eye(5), n_worker//5)
    B3[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta34, 0.5 *np.eye(5), n_worker//5)
    B3[4*n_worker//5: , :] = np.random.multivariate_normal(beta35, 0.5 *np.eye(5), n_worker//5)
    
    B4[:n_worker//5, :] = np.random.multivariate_normal(beta41, 0.5 *np.eye(5), n_worker//5)
    B4[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta42, 0.5 *np.eye(5), n_worker//5)
    B4[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta43, 0.5 *np.eye(5), n_worker//5)
    B4[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta44, 0.5 *np.eye(5), n_worker//5)
    B4[4*n_worker//5: , :] = np.random.multivariate_normal(beta45, 0.5 *np.eye(5), n_worker//5)
    
    B5[:n_worker//5, :] = np.random.multivariate_normal(beta51, 0.5 *np.eye(5), n_worker//5)
    B5[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta52, 0.5 *np.eye(5), n_worker//5)
    B5[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta53, 0.5 *np.eye(5), n_worker//5)
    B5[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta54, 0.5 *np.eye(5), n_worker//5)
    B5[4*n_worker//5: , :] = np.random.multivariate_normal(beta55, 0.5 *np.eye(5), n_worker//5)
    '''
    A[:n_task//5, :] = np.random.multivariate_normal(alpha_0, 0.1 * np.eye(5), n_task//5)
    A[n_task//5: 2*n_task//5, :] = np.random.multivariate_normal(alpha_1, 0.1 * np.eye(5), n_task//5)
    A[2*n_task//5: 3*n_task//5, :] = np.random.multivariate_normal(alpha_2, 0.1 *np.eye(5), n_task//5)
    A[3*n_task//5: 4*n_task//5, :] = np.random.multivariate_normal(alpha_3, 0.1 * np.eye(5), n_task//5)
    A[4*n_task//5:, :] = np.random.multivariate_normal(alpha_4, 0.1 * np.eye(5), n_task//5)

    B1[:n_worker//5, :] = np.random.multivariate_normal(beta11, 0.1 *np.eye(5), n_worker//5)
    B1[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta12, 0.1 *np.eye(5), n_worker//5)
    B1[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta13, 0.1 *np.eye(5), n_worker//5)
    B1[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta14,0.1 *np.eye(5), n_worker//5)
    B1[4*n_worker//5: , :] = np.random.multivariate_normal(beta15, 0.1 *np.eye(5), n_worker//5)
    
    B2[:n_worker//5, :] = np.random.multivariate_normal(beta21, 0.1 *np.eye(5), n_worker//5)
    B2[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta22, 0.1 *np.eye(5), n_worker//5)
    B2[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta23, 0.1 *np.eye(5), n_worker//5)
    B2[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta24, 0.1 *np.eye(5), n_worker//5)
    B2[4*n_worker//5: , :] = np.random.multivariate_normal(beta25, 0.1 *np.eye(5), n_worker//5)
    
    B3[:n_worker//5, :] = np.random.multivariate_normal(beta31, 0.1 *np.eye(5), n_worker//5)
    B3[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta32,0.1 * np.eye(5), n_worker//5)
    B3[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta33, 0.1 *np.eye(5), n_worker//5)
    B3[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta34, 0.1 *np.eye(5), n_worker//5)
    B3[4*n_worker//5: , :] = np.random.multivariate_normal(beta35, 0.1 *np.eye(5), n_worker//5)
    
    B4[:n_worker//5, :] = np.random.multivariate_normal(beta41, 0.1 *np.eye(5), n_worker//5)
    B4[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta42, 0.1 *np.eye(5), n_worker//5)
    B4[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta43, 0.1 *np.eye(5), n_worker//5)
    B4[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta44, 0.1 *np.eye(5), n_worker//5)
    B4[4*n_worker//5: , :] = np.random.multivariate_normal(beta45, 0.1 *np.eye(5), n_worker//5)
    
    B5[:n_worker//5, :] = np.random.multivariate_normal(beta51, 0.1 *np.eye(5), n_worker//5)
    B5[n_worker//5: 2*n_worker//5, :] = np.random.multivariate_normal(beta52, 0.1 *np.eye(5), n_worker//5)
    B5[2*n_worker//5: 3*n_worker//5, :] = np.random.multivariate_normal(beta53, 0.1 *np.eye(5), n_worker//5)
    B5[3*n_worker//5: 4*n_worker//5, :] = np.random.multivariate_normal(beta54, 0.1 *np.eye(5), n_worker//5)
    B5[4*n_worker//5: , :] = np.random.multivariate_normal(beta55, 0.1 *np.eye(5), n_worker//5)
    
    R_tsr = np.zeros((n_task, n_worker, 5))
    R_tsr[:, :, 0] = A.dot(B1.T)
    R_tsr[:, :, 1] = A.dot(B2.T)
    R_tsr[:, :, 2] = A.dot(B3.T)
    R_tsr[:, :, 3] = A.dot(B4.T)
    R_tsr[:, :, 4] = A.dot(B5.T)
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
    
    label = np.array([0] * (n_task // 5) + [1] * (n_task // 5) + [2] * (n_task // 5)+ [3] * (n_task // 5)+ [4] * (n_task // 5))
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
    l=[]
    for s in range(3):
        np.random.seed(s) # for the purpose of reproducibility, fix the seed
        model._mc_fit(rating, key = label, epsilon=1e-4, maxiter=50, verbose=0)
        
        candidate, assignment, pred_label, U = model._mc_infer(rating, label)
        group_acc = model.label_swap(U, label)
        tmp.append(np.mean(pred_label[:, 1] == label))
        tmp.append(np.mean(group_acc == label))
    #acc[seed] = np.max(tmp)
    
    
#print("---- accuracy: {0}({1}) ----".format(np.mean(acc), np.std(acc)))