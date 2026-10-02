# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:19:38 2025

@author: wangl
"""
from src.dawid_skene_model import DawidSkeneModel
import numpy as np
import pandas as pd
import numpy_indexed as npi
from sklearn.cluster import KMeans
from itertools import permutations
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import mode
from collections import Counter

class LFGP():
    def __init__(self, lf_dim=3, n_worker_group=2, lambda1=1, lambda2_0=1, lambda2_1=1):

        # Specify hyper-parameters

        self.lf_dim = lf_dim                    # dimension of latent factors     # number of worker subgroups
        self.lambda1 = lambda1                  # penalty coefficient for task subgrouping
        self.lambda2_0 = lambda2_0                  # penalty coefficient for worker subgrouping
        self.lambda2_1 = lambda2_1 
        
    def _prescreen(self, data):

        # fetch information from crowdsourced data
        n_task = len(np.unique(data[:, 0]))         # number of tasks
        n_worker = len(np.unique(data[:, 1]))       # number of workers
        n_task_group =  len(np.unique(data[:, 2]))  # number of task categories
        n_record = len(data[:, 0])                  # number of crowdsourced labels

        self.n_task = n_task
        self.n_worker = n_worker
        self.n_task_group = n_task_group  
        self.n_record = n_record
        
    def data_converter(self,data):

        n_task = len(np.unique(data[:, 0]))
        n_worker = len(np.unique(data[:, 1]))
        num_class = len(np.unique(data[:, 2]))
    
        data_tensor = np.zeros((n_task, n_worker, num_class))
    
        for row in data:
    
            data_tensor[int(row[0]), int(row[1]), int(row[2])] += 1
    
        return data_tensor
        
    def _init_task_member_ds(self, data):

        data_tensor = self.data_converter(data)
        model = DawidSkeneModel(self.n_task_group, max_iter=50, tolerance=10e-100)
        _, _, _, pred_label = model.run(data_tensor)

        label = np.zeros((self.n_task, 2))
        task = np.unique(data[:, 0]) # task index
        label[:, 0] = task
        label[:, 1] = np.argmax(pred_label, axis=1).squeeze()
        

        return label
    
    def _init_worker_member_acc(self, data, label):
        worker = np.unique(data[:, 1])  # Worker index
        acc = np.zeros((self.n_worker, self.n_task_group))  # Accuracy per task group
        member = np.zeros((self.n_worker, self.n_task_group), dtype=int)  # Worker groups
    
        for t_group in range(self.n_task_group):
            for i in range(self.n_worker):
                # Get data for the worker in this task group
                crowd_w = data[
                    (data[:, 1] == worker[i]) & 
                    (label[data[:, 0].astype(int), 1] == t_group)
                ]
                if crowd_w.shape[0] > 0:
                    task_w = crowd_w[:, 0]
                    acc[i, t_group] = (
                        sum(label[np.isin(label[:, 0], task_w), 1] == crowd_w[:, 2])
                        / crowd_w.shape[0]
                    )
                else:
                    # If no data, assign lowest accuracy
                    acc[i, t_group] = 0

        for t_group in range(self.n_task_group):
            median_acc = np.median(acc[:, t_group])
            # Group 1 if above median, else Group 0
            member[:, t_group] = (acc[:, t_group] > median_acc).astype(int)
    
        return member


    
    def _init_mc_params(self, data, scheme):

        # initialize model parameters for multicategory crowdsourcing
        # two initialization schemes are available: mv and random

        if scheme == "mv":

            task_member = self._init_task_member_mv(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member

            #if len(np.unique(V[:, 0])) < self.n_worker_group:
                #worker_member = self._init_worker_member_random(data)
                #V = worker_member[:, 1]

            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(worker_member)
            
        elif scheme == "ds":

            task_member = self._init_task_member_ds(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member

            #if len(np.unique(V)) < self.n_worker_group:
                #worker_member = self._init_worker_member_random(data)
                #V = worker_member[:, 1]

            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(worker_member)
        U = U.astype(int)
        V = V.astype(int)

        self.A, self.B = A, B
        self.U, self.V = U, V
        
    def _init_task_lf_gp(self, label):

        # initialize model parameters (task latent factors) using surrogate group information

        lf = np.zeros((self.n_task, self.lf_dim))
        for i in range(self.n_task_group):

            task_idx = label[label[:, 1] == i, 0].astype(int)
            tmp_centroid = 2 * np.random.rand(self.lf_dim) - 1
            #tmp_centroid = tmp_centroid / np.linalg.norm(tmp_centroid)
            lf[task_idx, :] = np.random.multivariate_normal(tmp_centroid, 0.2 * np.eye(self.lf_dim), len(task_idx))

        return lf
    

    def _init_worker_lf_gp(self, member):

        # initialize model parameters (worker latent factors) using surrogate group information

        lf = np.zeros((self.n_worker, self.n_task_group, self.lf_dim))

        for t_group in range(self.n_task_group):
            for i in range(self.n_worker):
                worker_idx = member[i, t_group]  # The worker's subgroup for this task group
                
                # Generate a latent factor for the worker in this task group
                tmp_centroid = 2 * np.random.rand(self.lf_dim) - 1  # Random initialization
                #tmp_centroid /= np.linalg.norm(tmp_centroid)  # Normalize
                lf[i, t_group, :] = np.random.multivariate_normal(tmp_centroid, 0.2 * np.eye(self.lf_dim))
    
        return lf  # Shape (n_worker, n_task_group, lf_dim)
    
    def _mc_loss_func(self, data, clusters):

        loss = 0
        _, task_id = np.unique(data[:, 0], return_inverse=True)  # Task indices
        _, worker_id = np.unique(data[:, 1], return_inverse=True)  # Worker indices
    
        for i in range(self.n_record):
            task_idx = task_id[i]
            worker_idx = worker_id[i]
    
            prob = [self.A[task_idx, :].dot(self.B[worker_idx, cla, :]) for cla in np.arange(self.n_task_group)]
            prob = np.divide(np.exp(prob), np.sum(np.exp(prob)))
    
            loss += -np.log(prob[int(data[i, 2])])
    
        # Compute penalties to enforce grouping structure
        gA, alpha = npi.group_by(self.U).mean(self.A, axis=0)  # Task group centroids
        beta = np.zeros((2, self.lf_dim, self.n_task_group))
        beta = clusters
        '''for t_group in range(self.n_task_group):
            group_labels = self.V[:, t_group]       # shape (n_worker,)
            B_slice = self.B[:, t_group, :]         # shape (n_worker, lf_dim)
    
            # group_by(...) finds each unique label & mean of B_slice for that label
            unique_groups, centroids = npi.group_by(group_labels).mean(B_slice, axis=0)
    
            # Overwrite centroid for any group == 0
            for i, g in enumerate(unique_groups):
                if g == 0:
                    # Force group 0's centroid to be zero:
                    centroids[i, :] = 0.0
    
            # Store computed centroids
            for i, g in enumerate(unique_groups):
                beta[g, t_group, :] = centroids[i, :]
'''

        penalty1 = self.lambda1 * np.sum([np.linalg.norm(self.A[self.U == p, :] - alpha[gA == p, :]) ** 2 for p in np.unique(self.U)])
        penalty2 = 0
        for j in range(self.n_task_group):
            mask0 = (self.V[:, j] == 0)
            mask1 = (self.V[:, j] == 1)
            diff = np.linalg.norm(self.B[mask0, j, :] - 0)
            penalty2 = penalty2 + self.lambda2_0*diff
            diff = np.linalg.norm(self.B[mask1, j, :] - beta[1, :, j])
            penalty2 = penalty2 + self.lambda2_1*diff
        '''  
        for j in range(self.n_task_group):
            for g in range(2):
                mask = (self.V[:, j] == g)
                if not np.any(mask):
                    continue
                
                diff = np.linalg.norm(self.B[mask, j, :] - beta[g, j,:])
                penalty2 += diff
        penalty2 *= self.lambda2
        '''
        
        return loss + penalty1 + penalty2
    
    def _comp_centroid(self, A, B, U, V):
        # Compute centroids for task groups
        group_A, centroid_A = npi.group_by(U).mean(A, axis=0)
        
        Centroid_A = np.zeros(A.shape)
        Centroid_B = np.zeros(B.shape)
        for g in range(self.n_task_group):
            Centroid_A[U == group_A[g], :] = centroid_A[g, :]
        
        for t_group in range(self.n_task_group):
            group_B, centroid_B = npi.group_by(V[:, t_group]).mean(B[:, t_group, :], axis=0)
    
            # Assign each worker the centroid of its group within this task group
            for g in np.unique(V[:, t_group]):
                Centroid_B[V[:, t_group] == g, t_group, :] = centroid_B[group_B == g, :]
    
        return Centroid_A, Centroid_B
    
    def multinomial_reg1(self,A_init, B, Y, worker_idx_array, lambd, centroid,data,task_idx):
        N = B.shape[0]
        k = self.lf_dim
        C = self.n_task_group
    
        conc1 = np.zeros((N, C))
        conc2 = np.zeros((N, C, k))
        beta = A_init
        for n in range(N):

            for c in range(C):
                
                conc1[n, c] = np.dot(beta.T, B[n, c, :])
                conc2[n, c, :] = B[n, c, :]
        conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
        grad = np.sum(- conc2[np.arange(N), Y.astype('int'), :] + \
                np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                2 * lambd * (beta - centroid)
        
        iter = 0
        #lossmem = np.zeros(11)
        while np.linalg.norm(grad) > 1e-1:
    
            beta = beta - 0.001 * grad
            for n in range(N):   #n_worker
        
                for c in range(C):    #n_task_group
                    
                    conc1[n, c] = np.dot(beta.T,  B[n, c, :])
                    conc2[n, c, :] =  B[n, c, :]
    
            conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
            grad = np.sum(- conc2[np.arange(N), Y.astype('int'), :] + \
                    np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                    2 * lambd * (beta - centroid)

            iter += 1
            if iter > 10:
                break
        return beta
    
    def multinomial_reg2(self,B_init, A, Y, lambd, centroid,data,worker_idx):
    
        M = A.shape[0]# n_task
        k = A.shape[1]#dim
        C = self.n_task_group

        conc1 = np.zeros((M, C))
        grad = np.zeros_like(B_init)
        beta = B_init
    
        for m in range(M):
            for c in range(C):
                conc1[m, c] = np.dot(A[m, :], beta[c, :])  # Logit for task m and class c
        conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1, keepdims=True)  # Softmax normalization

        for m in range(M):  # Iterate over tasks
            for c in range(C):  # Iterate over classes
                grad[c, :] += A[m, :] * (conc1[m, c] - (1 if c == Y[m] else 0))
        
        iter = 0
        while np.linalg.norm(grad) > 1e-1:
            beta -= 0.001 * grad
            for m in range(M):
                for c in range(C):
                    conc1[m, c] = np.dot(A[m, :], beta[c, :])  # Logit for task m and class c
            conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1, keepdims=True)  # Softmax normalization

            for m in range(M):  # Iterate over tasks
                for c in range(C):  # Iterate over classes
                    grad[c, :] += A[m, :] * (conc1[m, c] - (1 if c == Y[m] else 0))
                    
            grad += 2 * lambd * (beta - centroid)
            
            iter += 1
            if iter > 10:
                break
        #beta = beta / np.linalg.norm(beta, axis = 1, keepdims = True)
        return beta
    
    def label_swap(self,Grp_cur, Grp_prev):

        grp = np.unique(Grp_cur)
    
        perm_all = list(permutations(grp))
    
        rand_index_list = np.zeros((len(perm_all), ))
    
        for i in range(len(perm_all)):
    
            dic = {idx: perm_all[i][idx] for idx in grp}
            Grp_perm = [dic[Grp_cur[j]] for j in range(len(Grp_cur))]
    
            rand_index_list[i] = accuracy_score(Grp_prev, Grp_perm)
    
        ix = np.argmax(rand_index_list)
        dic = {idx: perm_all[ix][idx] for idx in grp}
        Grp_perm = [dic[Grp_cur[j]] for j in range(len(Grp_cur))]
    
        return np.array(Grp_perm)
    
    def new_kmeans(self, X):
        lf_dim = X.shape[1]
       
    
        # Initialize cluster centers
        cluster_centers = np.zeros((2, lf_dim))  # Center 0 is fixed at [0, 0, ..., 0]
        #cluster_centers[1] = X[np.random.choice(self.n_worker, 1, replace=False)]  # Random initialization for the second center
        cluster_centers[1] = X[np.argmax(np.linalg.norm(X, axis=1))]
        labels = np.zeros(self.n_worker)

        for i in range(300):
            distances = np.linalg.norm(X[:, np.newaxis] - cluster_centers, axis=2)
            labels = np.argmin(distances, axis=1)  # Label 0 for the fixed center, 1 for the other center
           

    
            new_centers = cluster_centers.copy()
            points_in_cluster_1 = X[labels == 1]
            if len(points_in_cluster_1) > 0:  # Avoid empty cluster
                new_centers[1] = points_in_cluster_1.mean(axis=0)
    
            if np.all(np.abs(new_centers - cluster_centers) < 1e-4):
                break
    
            cluster_centers = new_centers
            
            if np.linalg.norm(cluster_centers[1]) < np.linalg.norm(cluster_centers[0]):
                print(cluster_centers)
                labels = 1 - labels  # Swap labels if necessary
    
        return labels, cluster_centers


  
    
    def _mc_fit(self, oracle, data, key, scheme="ds",maxiter=50, epsilon=1e-5, verbose=0):
        self._init_mc_params(data, scheme=scheme)  # Initializes A, B, U, V
        task_ids, task_idx = np.unique(data[:, 0], return_inverse=True)  # Task indices
        worker_ids, worker_idx = np.unique(data[:, 1], return_inverse=True)  # Worker indices
        clusters = np.zeros((2, self.lf_dim, self.n_task_group))
        
        # Loss tracking
        loss_history = []
        loss_prev = float("inf")
        loss = np.zeros(maxiter)
        
        if verbose > 0:
            print("Starting optimization...")
            
        U_cur, V_cur = self.U, self.V
    
        for iter_count in range(maxiter):
            if verbose > 0:
                print(f"\nIteration {iter_count + 1}/{maxiter}")
    
            A_prev, B_prev = self.A.copy(), self.B.copy()
            U_prev, V_prev = self.U.copy(), self.V.copy()
            Alpha, Beta = self._comp_centroid(A_prev, B_prev, U_prev, V_prev)
    
            for t in range(self.n_task):
                # Observations related to task t
                obs_idx = np.where(data[:, 0] == task_ids[t])[0]
                obs_workers = worker_idx[obs_idx]
                obs_labels = data[obs_idx, 2].astype(int)
    

                task_group = self.U[t]
    
                # Update A[t, :]
                if len(obs_idx) > 0:
                    self.A[t, :] = self.multinomial_reg1(
                        A_init=self.A[t, :],
                        B=B_prev[obs_workers, :, :],
                        Y=obs_labels,
                        worker_idx_array=obs_workers,
                        lambd=self.lambda1,
                        centroid=Alpha[t, :],
                        data=data,
                        task_idx=t)
    
            for w in range(self.n_worker):
                for group in range(self.n_task_group):
                    # Tasks this worker participated in where they're in this group
                    obs_idx = np.where((data[:, 1] == worker_ids[w]) & (self.U[data[:, 0].astype(int)] == group))[0]
                    obs_tasks = task_idx[obs_idx]
                    obs_labels = data[obs_idx, 2].astype(int)
                    
                    worker_group = self.V[w, group]
                    
                    if worker_group == 1:
                        lambd = self.lambda2_1
                    else:
                        lambd = self.lambda2_0
    
                    # Collect relevant task factors
                    A_array = self.A[obs_tasks, :]
    
                    # Compute worker group centroid
                    
                    if iter_count==0:
                        centroid = np.mean(self.B[w, self.V[w, :] == group, :], axis=0)
                    else:
                        centroid = clusters[worker_group, :, group]
                                            
                    # Update B[w, group, :]
                    if len(obs_idx) > 0:
                        self.B[w, :, :] = self.multinomial_reg2(
                            B_init=self.B[w, :, :],
                            A=A_array,
                            Y=obs_labels,
                            lambd=lambd,
                            centroid=centroid,
                            data=data,
                            worker_idx = w
                        )
    
            U_cur = KMeans(n_clusters=self.n_task_group).fit_predict(self.A)
            U_cur = self.label_swap(U_cur, U_prev)
            
            
            for t in range(oracle.shape[0]):
                if oracle[t, 0] != oracle[t, 1]:
                    B_slice_0 = self.B[:, oracle[t, 0], :]
                    B_slice_1 = self.B[:, oracle[t, 1], :]
                    B_slice = np.hstack((B_slice_0, B_slice_1))
                V_cur[:, oracle[t, 0]], cluster_temp = self.new_kmeans(B_slice)
                clusters[:, :, oracle[t, 0]] = cluster_temp[:, :self.lf_dim]
                clusters[:, :, oracle[t, 1]] = cluster_temp[:, self.lf_dim:]
                V_cur[:, oracle[t, 1]] = V_cur[:, oracle[t, 0]]
                V_cur[:, oracle[t, 0]] = self.label_swap(V_cur[:, oracle[t, 0]], V_prev[:, oracle[t, 0]])
                V_cur[:, oracle[t, 1]] = self.label_swap(V_cur[:, oracle[t, 1]], V_prev[:, oracle[t, 1]])
                
            self.U, self.V = U_cur, V_cur
    
            loss_cur = self._mc_loss_func(data, clusters)
            loss_history.append(loss_cur)
    
            if verbose > 0:
                print(f"Loss: {loss_cur:.6f}, Change: {(loss_prev - loss_cur) / abs(loss_prev):.6e}")
    
            if abs(loss_prev - loss_cur) / abs(loss_prev) < epsilon:
                if verbose > 0:
                    print("Convergence achieved.")
                break
    
            loss_prev = loss_cur
            loss[iter_count] = loss_cur

        plt.plot(loss)
        plt.show()
    
        if verbose > 0:
            print("Optimization complete.")
    
        # Return the final model parameters and loss history
        return self.A, self.B, self.U, self.V, clusters
    
    def calculate_worker_accuracy(self, worker_label):

        worker_accuracy = np.zeros((self.n_task_group, 4))
        for t in range(self.n_task_group):
            worker_accuracy[t, 0] = np.mean(self.V[:, t] == worker_label[:, t])
            FP = np.sum((worker_label[:, t] == 0) & (self.V[:, t] == 1))
            TN = np.sum((worker_label[:, t] == 0) & (self.V[:, t] == 0))
            TP = np.sum((worker_label[:, t] == 1) & (self.V[:, t] == 1))
            FN = np.sum((worker_label[:, t] == 1) & (self.V[:, t] == 0))
            worker_accuracy[t, 1] = FP/(FP+TN) #False Positive
            worker_accuracy[t, 2] = TP/(TP+FN) #True Postive
            worker_accuracy[t, 3] = TP/(TP+FP) #Precision
        return worker_accuracy
            
    def _mc_infer(self, data):
        new_U = np.zeros(self.U.shape) - 1
        for t in range(self.n_task_group):
            hq_worker = np.where(self.V[:, t] == 1)[0]
            task_t = np.where(self.U == t)
            task_group_data = data[np.isin(data[:, 1], hq_worker) & np.isin(data[:, 0], task_t)]

            if task_group_data.shape[0] > 0:
                # Retrieve labels given by workers in group 1
                labels = task_group_data[:, 2]
    
                # Compute the majority label
                majority_label = mode(labels, axis=None).mode
            else:
                # If no workers in group 1 assigned labels, return None
                majority_label = None
            new_U[task_t] = majority_label 
    
        return new_U
    
    def _mc_infer_by_task(self, data):
        new_U = np.zeros(self.U.shape) - 1
        for t in range(self.n_task):
            task_t = self.U[t]
            hq_worker = np.where(self.V[:, task_t] == 1)[0]
            
            task_data = data[np.isin(data[:, 1], hq_worker) & (data[:, 0] == t)]

            if task_data.shape[0] > 0:
                # Retrieve labels given by workers in group 1
                labels = task_data[:, 2]
    
                # Compute the majority label
                majority_label = mode(labels, axis=None).mode
            else:
                # If no workers in group 1 assigned labels, return None
                majority_label = None
            new_U[t] = majority_label 
    
        return new_U
    
    def diagnosis(self, worker_label, data):
        
        for t in range(self.n_task_group):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            wrong_hq_worker = np.where((self.V[:, t] == 1) & (worker_label[:, t] == 0))[0]
            correct_hq_worker = np.where((self.V[:, t] == 1) & (worker_label[:, t] == 1))[0]
            task_labeled_t = np.where(self.U == t)
            wrong_labels = np.where(np.isin(data[:, 0], task_labeled_t) & np.isin(data[:, 1], wrong_hq_worker))
            correct_labels = np.where(np.isin(data[:, 0], task_labeled_t) & np.isin(data[:, 1], correct_hq_worker))
            axes[0].hist(data[wrong_labels, 2].flatten(), bins = self.n_task_group, alpha=0.5, label=f'False hq {t+1}', edgecolor='black')
            axes[1].hist(data[correct_labels, 2].flatten(), bins = self.n_task_group, alpha=0.5, label=f'True hq {t+1}', edgecolor='black')
            plt.tight_layout()  # Adjusts spacing
            plt.show()
    def distance_graph(self, centers, worker_label):
        for t in range(self.n_task_group):
            V_tmp = worker_label[:, t]
            B_tmp = self.B[:, t, :]       
            B0 = B_tmp[V_tmp == 0]
            B1 = B_tmp[V_tmp == 1]
            distance0_0 = np.linalg.norm(B0 - centers[0, :, t], axis=1)
            distance0_1 = np.linalg.norm(B0 - centers[1, :, t], axis=1)
            distance1_0 = np.linalg.norm(B1 - centers[0, :, t], axis=1)
            distance1_1 = np.linalg.norm(B1 - centers[1, :, t], axis=1)
            plt.scatter(distance0_0, distance0_1)
            plt.scatter(distance1_0, distance1_1)
            plt.show()
     
    def accuracy_worker(self, data, key):
        new_U = self.label_swap(self.U, key)
        worker_acc = np.zeros((self.n_worker, self.n_task_group))
        for t in range(self.n_task_group):
            task_list = np.where(new_U == t)[0]
            worker_acc_for_t = np.zeros(self.n_worker)
            for w in range(self.n_worker):
                task_worker_data = data[(data[:, 1] == w) & (np.isin(data[:, 0], task_list))]
                worker_acc_for_t[w] = np.mean(task_worker_data[:,2] == t)
            worker_acc[:, t] = worker_acc_for_t
            
        return worker_acc
    
    def _mc_infer_top2(self, data, key):

        top2_U = [[] for _ in range(self.n_task)]
        proportions = np.zeros((self.n_task_group, 2))
        
        for t in range(self.n_task_group):
            top2_label = np.zeros(2)
            top2_props = np.zeros(2)
            hq_worker = np.where(self.V[:, t] == 1)[0]
            task_t_indices = np.where(self.U == t)[0]
            task_group_data = data[np.isin(data[:, 1], hq_worker) & np.isin(data[:, 0], task_t_indices)]

            if task_group_data.shape[0] > 0:
                labels = task_group_data[:, 2]
                counter = Counter(labels)
                top2 = counter.most_common(2)
                i = 0
                for label,count in top2:
                    top2_label[i] = int(label)
                    top2_props[i] = count / len(task_group_data)
                    i = i + 1

            else:
                top2_label = []
    
            for idx in task_t_indices:
                top2_U[idx] = top2_label
                
            proportions[t, :] = top2_props
                
        correct = 0        
        
        for pred, true in zip(top2_U, key):
            if true in pred:
                correct += 1

        proportions_plot = proportions

        for i in range(len(proportions)):
            proportions_plot[i, 0] = max(proportions[i, 0], proportions[i, 1])
            proportions_plot[i, 1] = min(proportions[i, 0], proportions[i, 1])
            
        task = range(self.n_task_group)
        plt.bar(task, proportions_plot[:, 0], color = 'r')
        plt.bar(task, proportions_plot[:, 1], bottom = proportions_plot[:, 0],color = 'b')
        plt.show()
        return correct / len(key), top2_U, proportions

    def _mc_infer_by_task_top2(self, data, key):

        top2_U = [[] for _ in range(self.n_task)]
        proportions = np.zeros((self.n_task, 2))
        
        for t in range(self.n_task):
            task_t = self.U[t]
            top2_label = np.zeros(2)
            top2_props = np.zeros(2)
            hq_worker = np.where(self.V[:, task_t] == 1)[0]
            task_data = data[np.isin(data[:, 1], hq_worker) & (data[:, 0] == t)]
    
            if task_data.shape[0] > 0:
                labels = task_data[:, 2]
                counter = Counter(labels)
                top2 = counter.most_common(2)
                i = 0
                for label,count in top2:
                    top2_label[i] = int(label)
                    top2_props[i] = count / len(task_data)
                    i = i + 1
                
            else:
                top2_label = []
                
            top2_U[t] = top2_label
            proportions[t, :] = top2_props
                
        correct = 0
        
        for pred, true in zip(top2_U, key):
            if true in pred:
                correct += 1
                
        proportions_plot = proportions
        
        for i in range(len(proportions)):
            proportions_plot[i, 0] = max(proportions[i, 0], proportions[i, 1])
            proportions_plot[i, 1] = min(proportions[i, 0], proportions[i, 1])
        task = range(self.n_task)
        plt.bar(task, proportions_plot[:, 0], color = 'r')
        plt.bar(task, proportions_plot[:, 1], bottom = proportions_plot[:, 0],color = 'b')
        plt.show()
        return correct/len(key), top2_U, proportions
    
    def proportions(self, U_mv_by_task, key, worker_key):
        diff_prop = np.zeros((self.n_task, 2))
        for t in range(self.n_task):
            #task_t = int(U_mv_by_task[t])#determine task group
            task_t = int(key[t])
            hq_worker_pred = self.V[:, task_t]
            hq_worker_key = worker_key[:, task_t]

            #the proportion of true high quality workers in the identified hq
            diff_prop[t, 0] = np.sum((hq_worker_pred==1) & (hq_worker_key==1))/np.sum(hq_worker_pred==1)
            #out of the true labels how many are identified
            diff_prop[t, 1] = np.sum((hq_worker_pred==1) & (hq_worker_key==1))/np.sum(hq_worker_key==1)
        task = range(self.n_task)
        plt.bar(task, diff_prop[:, 0], color = 'b')
        plt.show()
        plt.bar(task, diff_prop[:, 1], color = 'r')
        plt.show()
        return diff_prop
                
            
                
            