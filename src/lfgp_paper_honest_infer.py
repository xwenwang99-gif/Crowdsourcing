# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:26:43 2024

@author: wangl
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from itertools import permutations
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy_indexed as npi
from sklearn.mixture import GaussianMixture
import copy
from src.dawid_skene_model import DawidSkeneModel
import logging
from scipy.stats import mode
from collections import Counter

import numpy as np

class LFGP():
    def __init__(self, lf_dim=3, n_worker_group=3, lambda1=1, lambda2=1):

        # Specify hyper-parameters

        self.lf_dim = lf_dim                    # dimension of latent factors
        self.n_worker_group= n_worker_group     # number of worker subgroups
        self.lambda1 = lambda1                  # penalty coefficient for task subgrouping
        self.lambda2 = lambda2                  # penalty coefficient for worker subgrouping

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
        
    def multinomial_reg1(self,A_init, B, Y, O, lambd, centroid,data,dex):

        N = B.shape[0]
        k = B.shape[1]
        C = O.shape[0]
    
        conc1 = np.zeros((N, C))
        conc2 = np.zeros((N, C, k))
        beta = A_init
        for n in range(N):
    
            for c in range(C):
                
                conc1[n, c] = np.dot(np.dot(beta.T, O[c, n, :, :]), B[n, :])
                conc2[n, c, :] = np.dot(O[c, n, :, :], B[n, :])
    
        conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
    
        grad = np.sum(- conc2[np.arange(N), Y.astype('int'), :] + \
                np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                2 * lambd * (beta - centroid)
        
        iter = 0
        #lossmem = np.zeros(11)
        while np.linalg.norm(grad) > 1e-1:
    
            beta = beta - 0.001 * grad
            for n in range(N):
        
                for c in range(C):
                    
                    conc1[n, c] = np.dot(np.dot(beta.T, O[c, n, :, :]), B[n, :])
                    conc2[n, c, :] = np.dot(O[c, n, :, :], B[n, :])
    
            conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
            grad = np.sum(- conc2[np.arange(N), Y.astype('int'), :] + \
                    np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                    2 * lambd * (beta - centroid)

            iter += 1
            if iter > 10:
                break
        return beta
    def multinomial_reg2(self,B_init, A, Y, O, lambd, centroid,data,dex):
    
        M = A.shape[0]
        k = A.shape[1]
        C = O.shape[0]
    
        conc1 = np.zeros((M, C))
        conc2 = np.zeros((M, C, k))
        #lossmem=np.zeros(1001)
        beta = B_init
    
        for m in range(M):
    
            for c in range(C):
                
                conc1[m, c] = np.dot(np.dot(A[m, :], O[c, :, :]), beta)
                conc2[m, c, :] = np.dot(O[c, :, :].T, A[m, :])
    
        conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
        
        grad = np.sum(- conc2[np.arange(M), Y.astype('int'), :] + \
                np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                2 * lambd * (beta - centroid)
        
        iter = 0
        while np.linalg.norm(grad) > 1e-1:
    
            beta = beta - 0.001 * grad
            
            for m in range(M):
        
                for c in range(C):
                    
                    conc1[m, c] = np.dot(np.dot(A[m, :], O[c, :, :]), beta)
                    conc2[m, c, :] = np.dot(O[c, :, :].T, A[m, :])
    
            conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
        
            grad = np.sum(- conc2[np.arange(M), Y.astype('int'), :] + \
                    np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                    2 * lambd * (beta - centroid)
            
            iter += 1
            if iter > 10:
                break
        return beta
    
    def cayley_transform(self,A, B, Y, O):

        G = np.zeros(O.shape)
        S = np.zeros(O.shape)
        C = G.shape[0]
        prob = np.zeros(Y.shape)
    
        for c in range(C):
    
            prob[:, c] = np.exp(np.sum(A * (B.dot(O[c, :, :].T)), axis=1))
        
        prob = prob / np.sum(prob, axis=1)[:, np.newaxis]
        coef = prob - Y
    
        iter = 0
        err = 1
    
        while err > 1e-2:
        
            for c in range(1, C): # the first set of rotation matrices are fixed as identity matrix for reference group
    
                G[c, :, :] = B.T.dot(np.diagflat(coef[:, c])).dot(A)
                S[c, :, :] = G[c, :, :].dot(O[c, :, :].T) - O[c, :, :].dot(G[c, :, :].T)
                O[c, :, :] = np.linalg.inv(np.eye(O.shape[1]) + 0.00005 * S[c, :, :]).dot(np.eye(O.shape[1]) - 0.00005 * S[c, :, :]).dot(O[c, :, :])
    
            for c in range(C):
        
                prob[:, c] = np.sum(np.exp(A.dot(O[c, :, :]) * B), axis=1)
    
            prob = prob / np.sum(prob, axis=1)[:, np.newaxis]
            coef = prob - Y
    
            err = np.max([np.linalg.norm(G[c, :, :]) for c in range(C)])
            iter += 1
    
            if iter > 10:
                break
            
    
        return O
    
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

    
    def _init_mc_params(self, data, scheme):

        # initialize model parameters for multicategory crowdsourcing
        # two initialization schemes are available: mv and random

        if scheme == "mv":

            task_member = self._init_task_member_mv(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member[:, 1]

            if len(np.unique(V)) < self.n_worker_group:
                worker_member = self._init_worker_member_random(data)
                V = worker_member[:, 1]

            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(worker_member)
            O = self._init_orth_rand()
            
        elif scheme == "ds":

            task_member = self._init_task_member_ds(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member[:, 1]

            if len(np.unique(V)) < self.n_worker_group:
                worker_member = self._init_worker_member_random(data)
                V = worker_member[:, 1]

            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(worker_member)
            O = self._init_orth_rand()
        U = U.astype(int)
        V = V.astype(int)

        self.A, self.B, self.O = A, B, O
        self.U, self.V = U, V
        
    def _init_task_member_mv(self, data):

        # initialize model parameters (task initial subgroup membership) using majority voting scheme

        task = np.unique(data[:, 0]) # task index
        label = np.zeros((self.n_task, 2))
        label[:, 0] = task

        for i in range(self.n_task):
            val, _ = stats.mode(data[data[:, 0] == task[i], 2]) # get the majority of crowdsourced label for each task
            label[i, 1] = val[0]                                # assign the majority voted label to the initial label

        return label

    def _init_worker_member_acc(self, data, label):

        # initialize model parameters (worker initial subgroup membership) by truncating the surrogate accuracy

        worker = np.unique(data[:, 1]) # worker index
        acc = np.zeros((self.n_worker, 2))
        acc[:, 0] = worker
        member = np.zeros((self.n_worker, 2))
        member[:, 0] = worker

        for i in range(self.n_worker):
            crowd_w = data[data[:, 1] == worker[i], :]
            task_w = crowd_w[:, 0]
            acc[i, 1] = sum(label[np.isin(label[:, 0], task_w), 1] == crowd_w[:, 2]) / crowd_w.shape[0]
        
        try:

            member[:, 1] = np.digitize(acc[:, 1], [np.min(acc[:, 1]) + (np.max(acc[:, 1]) + 0.01 - np.min(acc[:, 1])) * i / self.n_worker_group for i in range(1, self.n_worker_group + 1)], right=True)

        except:

            member[:, 1] = np.digitize(acc[:, 1], [np.quantile(acc[:, 1], i/self.n_worker_group) for i in range(1, self.n_worker_group + 1)], right=True)

        return member
    
    def _init_task_lf_gp(self, label):

        # initialize model parameters (task latent factors) using surrogate group information

        lf = np.zeros((self.n_task, self.lf_dim))
        for i in range(self.n_task_group):

            task_idx = label[label[:, 1] == i, 0].astype(int)
            tmp_centroid = 2 * np.random.rand(self.lf_dim) - 1
            tmp_centroid = tmp_centroid / np.linalg.norm(tmp_centroid)
            lf[task_idx, :] = np.random.multivariate_normal(tmp_centroid, 0.2 * np.eye(self.lf_dim), len(task_idx))

        return lf

    def _init_worker_lf_gp(self, member):

        # initialize model parameters (worker latent factors) using surrogate group information

        lf = np.zeros((self.n_worker, self.lf_dim))

        for i in range(self.n_worker_group):

            worker_idx = member[member[:, 1] == i, 0].astype(int)
            tmp_centroid = 2 * np.random.rand(self.lf_dim) - 1
            tmp_centroid = tmp_centroid / np.linalg.norm(tmp_centroid)
            lf[worker_idx, :] = np.random.multivariate_normal(tmp_centroid, 0.2 * np.eye(self.lf_dim), len(worker_idx))

        return lf
    
    def _init_orth_rand(self):

        # initialize orthogonal matrices for multi-categorical crowdsourcing with random orthogonal matrices

        O = np.zeros((self.n_task_group, self.n_worker_group, self.lf_dim, self.lf_dim))

        for i in range(self.n_task_group):
    
            for j in range(self.n_worker_group):
                
                if i == 0:
                    O[i, j, :, :] = np.eye(self.lf_dim)
                else:
                    O[i, j, :, :] = stats.ortho_group.rvs(self.lf_dim)
        return O
    def _comp_centroid(self, A, B, U, V):

        group_A, centroid_A = npi.group_by(U).mean(A, axis=0)
        group_B, centroid_B = npi.group_by(V).mean(B, axis=0)

        Centroid_A = np.zeros(A.shape)
        Centroid_B = np.zeros(B.shape)

        for g in range(self.n_task_group):
            Centroid_A[U == group_A[g], :] = centroid_A[g, :]
        for g in range(self.n_worker_group):
            Centroid_B[V == group_B[g], :] = centroid_B[g, :]

        return Centroid_A, Centroid_B  
    
    def _mc_loss_func(self, data):

        _, task_id = np.unique(data[:, 0], return_inverse=True)
        _, worker_id = np.unique(data[:, 1], return_inverse=True)

        loss = 0

        for i in range(self.n_record):
        
            prob = [self.A[task_id[i], :].dot(self.O[cla, self.V[worker_id[i]], :, :]).dot(self.B[worker_id[i], :]) for cla in np.arange(self.n_task_group)]
            prob = np.divide(np.exp(prob), np.sum(np.exp(prob)))

            loss += - np.log(prob[int(data[i, 2])])

        gA, alpha = npi.group_by(self.U).mean(self.A, axis=0)
        gB, beta = npi.group_by(self.V).mean(self.B, axis=0)

        penalty1 = self.lambda1 * np.sum([np.linalg.norm(self.A[self.U == p, :] - alpha[gA == p, :]) ** 2 for p in np.unique(self.U)])
        penalty2 = self.lambda2 * np.sum([np.linalg.norm(self.B[self.V == q, :] - beta[gB == q, :]) ** 2 for q in np.unique(self.V)])

        return loss + penalty1 + penalty2
    
    def _mc_fit(self, data, key, scheme="ds", maxiter=50, epsilon=1e-5, verbose=0):
        
        
        tmp = []
        
        task, task_id = np.unique(data[:, 0], return_inverse=True)
        worker, worker_id = np.unique(data[:, 1], return_inverse=True)

        self._init_mc_params(data, scheme=scheme)

        iter = 0
        loss_cur = self._mc_loss_func(data)
        err = 1

        if verbose > 0:
            print("Iter: {0}, loss: {1}".format(iter, loss_cur))

        A_cur, B_cur, O_cur = self.A, self.B, self.O
        U_cur, V_cur = self.U, self.V
        
        while err > epsilon:
            A_prev, B_prev, O_prev = A_cur, B_cur, O_cur
            U_prev, V_prev = U_cur, V_cur
            Alpha, Beta = self._comp_centroid(A_prev, B_prev, U_prev, V_prev)

            loss_prev = loss_cur

            iter = iter + 1

            # update A
            
            for id, _ in enumerate(task):

                obsIdx = np.argwhere(task_id == id).squeeze()
                obsWorker = worker_id[obsIdx]
                obsB = B_prev[obsWorker, :]
                obsY = data[obsIdx, 2]
                obsO = O_prev[:, V_prev[obsWorker], :, :]
                if obsIdx.size < 2:
                    obsB = obsB[np.nexaxis, :]
                    obsY = np.array([obsY])

                A_cur[id, :] = self.multinomial_reg1(A_prev[id, :], obsB, obsY, obsO, self.lambda1, Alpha[id, :].T,data,id)
            
            self.A = A_cur
            
            # update B
            for jd, _ in enumerate(worker):

                obsIdx = np.argwhere(worker_id == jd).squeeze()
                obsTask = task_id[obsIdx]
                obsA = A_cur[obsTask, :]
                obsY = data[obsIdx, 2]
                obsO = O_prev[:, V_prev[jd], :, :]
                if obsIdx.size < 2:
                    obsA = obsA[np.newaxis, :]
                    obsY = np.array([obsY])

                B_cur[jd, :] = self.multinomial_reg2(B_prev[jd, :], obsA, obsY, obsO, self.lambda2, Beta[jd, :].T,data,jd)
            self.B = B_cur
            
            # update O
            for jd in range(self.n_worker_group):
                
                obsIdx = np.argwhere(V_prev[worker_id] == jd).squeeze()
                obsTask = task_id[obsIdx]
                obsWorker = worker_id[obsIdx]
                obsA = A_cur[obsTask, :]
                obsB = B_cur[obsWorker, :]
                obsY = data[obsIdx, 2]
                obsY_dummy = np.zeros((len(obsY), self.n_task_group))
                obsY_dummy[np.arange(len(obsY)), obsY.astype('int')] = 1
                obsO = O_prev[:, jd, :, :]
                O_cur[:, jd, :, :] = self.cayley_transform(obsA, obsB, obsY_dummy, obsO)
            self.O = O_cur
            
            # update U
            U_cur = KMeans(n_clusters=self.n_task_group).fit_predict(A_cur)
            #U_cur = self.label_swap(U_cur, U_prev)

            # update V
            V_cur = KMeans(n_clusters=self.n_worker_group).fit_predict(B_cur)
            #V_cur = self.label_swap(V_cur, V_prev)


            self.A, self.B, self.O = A_cur, B_cur, O_cur
            self.U, self.V = U_cur, V_cur

            loss_cur = self._mc_loss_func(data)
            _, task_id = np.unique(data[:, 0], return_inverse=True)
            _, worker_id = np.unique(data[:, 1], return_inverse=True)

            loss = 0

            for i in range(self.n_record):
            
                prob = [self.A[task_id[i], :].dot(self.O[cla, self.V[worker_id[i]], :, :]).dot(self.B[worker_id[i], :]) for cla in np.arange(self.n_task_group)]
                prob = np.divide(np.exp(prob), np.sum(np.exp(prob)))
    
                loss += - np.log(prob[int(data[i, 2])])
            if verbose > 0:
                print("Iter: {0}, loss: {1}".format(iter, loss_cur))

            err = np.abs(loss_cur - loss_prev) / loss_prev

            if iter > maxiter:
                break

            self.iter = iter


        self.A, self.B, self.O, self.U, self.V = A_cur, B_cur, O_cur, U_cur, V_cur
        
        return self.A, self.B, self.U, self.V

    def _mc_honest_infer(self, data, key):
    
            gA, alpha = npi.group_by(self.U).mean(self.A, axis=0)
            gB, beta = npi.group_by(self.V).mean(self.B, axis=0)
    
            label = np.zeros((self.n_task, 2))
            label[:, 0] = np.unique(data[:, 0])
            
            
            assignment = np.zeros((self.n_task_group, 2))
            assignment[:, 0] = range(self.n_task_group)
            
            candidate = np.zeros((self.n_task_group, self.n_worker_group, len(gA)))
            
            for i in gA:
                candidate_temp = np.zeros((self.n_task_group, self.n_worker_group))
                for ii in range(self.n_task_group):
                    for jj in range(self.n_worker_group):
                        candidate_temp[ii, jj] = alpha[ii, :].dot(self.O[i, jj, :, :]).dot(beta[jj, :])
                candidate[:, :, i] = candidate_temp
                
                x = np.argwhere(candidate_temp == np.max(candidate_temp)) #assign label i to group ii
                assignment[i, 1] = x[0, 0]
                label[self.U == x[0,0], 1] = i               
    
    
            return label, assignment, alpha, beta
        
    def _mc_infer(self, data):
        new_U = np.zeros(self.U.shape) - 1
        for t in range(self.n_task_group):
            hq_worker = np.where(self.V == t)[0]
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
            hq_worker = np.where(self.V == t)[0]
            
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

