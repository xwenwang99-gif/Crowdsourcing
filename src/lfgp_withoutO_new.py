# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 01:00:27 2025

@author: wangl
"""

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

class LFGP():
    def __init__(self, lf_dim=3, n_worker_group=2, lambda1=1, lambda2=1):

        # Specify hyper-parameters

        self.lf_dim = lf_dim                    # dimension of latent factors     # number of worker subgroups
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
            tmp_centroid = tmp_centroid / np.linalg.norm(tmp_centroid)
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
                tmp_centroid /= np.linalg.norm(tmp_centroid)  # Normalize
                lf[i, t_group, :] = np.random.multivariate_normal(tmp_centroid, 0.2 * np.eye(self.lf_dim))
    
        return lf  # Shape (n_worker, n_task_group, lf_dim)
    
    def _mc_loss_func(self, data):

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
        beta = np.zeros((2, self.n_task_group, self.lf_dim))
        for t_group in range(self.n_task_group):
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


        penalty1 = self.lambda1 * np.sum([np.linalg.norm(self.A[self.U == p, :] - alpha[gA == p, :]) ** 2 for p in np.unique(self.U)])
        penalty2 = 0
        for j in range(self.n_task_group):
            for g in range(2):
                mask = (self.V[:, j] == g)
                if not np.any(mask):
                    continue
                
                diff = np.linalg.norm(self.B[mask, j, :] - beta[g, j,:])
                penalty2 += diff
        penalty2 *= self.lambda2
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
    
    import numpy as np

    def gradient_descent_A(self, A_init, B, Y, worker_idx_array, lambd, centroid, data, task_idx):
        # Initialize variables
        A = A_init.copy()  # Initialize with the given latent factor
        lf_dim = A.shape[0]  # Latent factor dimension
        N = len(worker_idx_array)  # Number of workers
        C = len(np.unique(Y))  # Number of classes
    
        for iteration in range(10):
            # Step 1: Compute gradient
            grad = np.zeros(lf_dim)
    
            # Loop over workers contributing to this task
            for n_idx, w in enumerate(worker_idx_array):
                group_w = self.V[w, task_idx]  # Worker group for this task group
    
                # Compute logits and softmax probabilities
                logits = np.array([np.dot(A, B[w, c, :]) for c in range(C)])  # shape (C,)
                probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax probabilities
    
                # True label
                true_label = Y[n_idx]
    
                # Gradient contribution from this worker
                for c in range(C):
                    grad += B[w, c, :] * (1.0 - probs[true_label])  # Contribution from true label
                    for c_prime in range(C):  # Contributions from all classes
                        if c_prime != true_label:
                            grad -= probs[c_prime] * worker_factor
                            grad += B[w, c, :] * (1.0 - probs[true_label]) 
    
            # Add regularization gradient
            grad -= 2 * lambd * (A - centroid)
    
            # Step 2: Update A
            A = A - 0.001 * grad  # Gradient descent update
    
            # Step 3: Check for convergence
            grad_norm = np.linalg.norm(grad)  # Norm of the gradient
            if grad_norm <  1e-1:
                print(f"Converged after {iteration+1} iterations with gradient norm {grad_norm:.4e}.")
                return A
    
        return A

    
    def multinomial_reg2(self,B_init, A, Y, lambd, centroid,data):

            # Number of tasks in this batch (M) and dimension (k)
        M, k = A.shape
        # Number of classes
        C = len(np.unique(Y))
    
        # conc1 stores logits, conc2 helps with gradient construction
        conc1 = np.zeros((M, C))       # shape: (M, C)
        conc2 = np.zeros((M, C, k))    # shape: (M, C, k)
    
        beta = B_init.copy()  # Start with the provided vector
    
        # 1) Compute initial logits conc1 and helper conc2
        for m in range(M):
            # For each class c, we get a "score" = A[m,:] dot beta
            for c in range(C):
                conc1[m, c] = np.dot(A[m, :], beta)
                # Each row in conc2 is just A[m, :] since derivative wrt beta is A[m, :]
                conc2[m, c, :] = A[m, :]
    
        # 2) Softmax normalization over classes
        #    For each m, exponentiate and divide by sum of exps
        conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1, keepdims=True)
    
        # 3) Compute gradient
        #    "Y.astype('int')" is used if Y is numeric classes 0..C-1
        grad = np.sum(
            -conc2[np.arange(M), Y.astype(int), :] +
            np.sum(conc1[:, :, np.newaxis] * conc2, axis=1),
            axis=0
        ) + 2 * lambd * (beta - centroid)
    
        # 4) Gradient Descent Update Loop
        iteration = 0
        step_size = 0.001  # Could tune if needed
        while np.linalg.norm(grad) > 1e-1:
            # Update beta
            beta -= step_size * grad
    
            # Recompute logits/grad
            for m in range(M):
                for c in range(C):
                    conc1[m, c] = np.dot(A[m, :], beta)
                    conc2[m, c, :] = A[m, :]
    
            conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1, keepdims=True)
    
            grad = np.sum(
                -conc2[np.arange(M), Y.astype(int), :] +
                np.sum(conc1[:, :, np.newaxis] * conc2, axis=1),
                axis=0
            ) + 2 * lambd * (beta - centroid)
    
            iteration += 1
            if iteration > 10:
                break  # Safety stop
    
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

  
    
    def _mc_fit(self, data, key, scheme="ds", maxiter=50, epsilon=1e-5, verbose=0):
    
        # 1) Initialize A, B, U, V (no O)
        self._init_mc_params(data, scheme=scheme)
        # Now we have:
        # self.A : (n_task, lf_dim)
        # self.B : (n_worker, n_task_group, lf_dim)
        # self.U : (n_task,)            - each task's group
        # self.V : (n_worker, n_task)   - each worker's group per task
    
        iter_count = 0
        loss_cur = self._mc_loss_func(data) 
        loss_prev = float('inf')
        err = 1.0
    
        loss_history = [loss_cur]
    
        if verbose > 0:
            print(f"Iter: {iter_count}, loss: {loss_cur:.4f}")
    
        # 2) Main Loop
        while err > epsilon and iter_count < maxiter:
            iter_count += 1
    
            # -----------------------------------------------------------
            # A) Compute Group Centroids for Regularization
            #    We'll do it once per iteration, and pass these into
            #    multinomial_reg1 / multinomial_reg2 for the penalty term.
            # -----------------------------------------------------------
            Centroid_A, Centroid_B = self._comp_centroid(self.A, self.B, self.U, self.V)
            # - Centroid_A : (n_task, lf_dim)      but actually each row is the centroid 
            #                for the group in U. We will pick the right row as needed.
            # - Centroid_B : (n_worker, n_task, lf_dim) or (n_worker_group, n_task_group, lf_dim),
            #                depending on your design. In your code, you store per-worker centroids, 
            #                but typically we'd do (n_worker_group, n_task_group, lf_dim). 
            #
            # Make sure your _comp_centroid returns the shape you expect.
            #
            # Alternatively, you can store your group centroids in a dictionary or 
            # compute them inside the reg functions.
    
            # -----------------------------------------------------------
            # B) Update Task Latent Factors (A)
            #    For each task, gather (worker, label) pairs, build B_array,
            #    call multinomial_reg1(...) to update A[task_idx, :].
            # -----------------------------------------------------------
            A_prev = self.A.copy()
            B_prev = self.B.copy()
            for task_idx in range(self.n_task):
                obs_idx = np.where(data[:, 0] == task_idx)[0]  # rows with this task
                if obs_idx.size == 0:
                    continue
    
                obs_workers = data[obs_idx, 1].astype(int)
                obs_labels = data[obs_idx, 2].astype(int)
                self.A[task_idx, :] = self.multinomial_reg1(
                    A_init=A_prev[task_idx, :],
                    B=B_prev,
                    Y=obs_labels,
                    worker_idx_array = obs_workers,
                    lambd=self.lambda1,
                    centroid=Centroid_A[task_idx, :],
                    data=data, 
                    task_idx=task_idx
                )
         
            
            # -----------------------------------------------------------
            # C) Update Worker Latent Factors (B)
            #    For each worker and each group_label in [0..n_task_group),
            #    gather tasks, build A_array, call multinomial_reg2(...) 
            # -----------------------------------------------------------
            
            for worker_idx in range(self.n_worker):
                for group_label in range(self.n_task_group):
                    # tasks assigned to this group_label for this worker
                    task_idx_array = np.where(self.V[worker_idx, :] == group_label)[0]
                    if task_idx_array.size == 0:
                        continue
    
                    # Collect row indices from data for those tasks
                    obs_id_list = []
                    for t_idx in task_idx_array:
                        obs_id_list.extend(np.where((data[:, 0] == t_idx) & (data[:, 1] == worker_idx))[0])
                    obs_id_list = np.array(obs_id_list, dtype=int)
                    if obs_id_list.size == 0:
                        continue
    
                    obs_labels = data[obs_id_list, 2].astype(int)
                    A_array = []
                    for row_id in obs_id_list:
                        t_idx = int(data[row_id, 0])
                        A_array.append(self.A[t_idx, :])
                    A_array = np.vstack(A_array) if len(A_array) > 0 else np.zeros((0, self.lf_dim))
    
                    # Retrieve centroid for (worker_idx, group_label) or 
                    # for (group_label) if you store them by group only.
                    # If _comp_centroid returns shape (n_worker, n_task, lf_dim), 
                    # then:
                    centroid_B = Centroid_B[worker_idx, group_label, :]
    
                    B_init = B_prev[worker_idx, group_label, :]
    
                    if A_array.shape[0] > 0:
                        self.B[worker_idx, group_label, :] = self.multinomial_reg2(
                            B_init=B_init,
                            A=A_array,
                            Y=obs_labels,
                            lambd=self.lambda2,
                            centroid=centroid_B,
                            data=data,
                            task_idx=-1  # Might not be needed
                        )
    
            # -----------------------------------------------------------
            # D) Re-cluster Tasks and Workers
            #    1) Cluster tasks to update U
            #    2) Cluster workers to update each column of V
            # -----------------------------------------------------------
            U_prev = self.U.copy()
            self.U = KMeans(n_clusters=self.n_task_group, n_init=10).fit_predict(self.A)
            self.U = self.label_swap(self.U, U_prev)
    
            V_prev = self.V.copy()
            # Option: reassign each task's column in V using KMeans on B
            for t_idx in range(self.n_task):
                # Flatten B for all worker_idx into shape (n_worker, n_task_group*lf_dim)
                B_flat = self.B.reshape(self.n_worker, self.n_task_group * self.lf_dim)
                self.V[:, t_idx] = KMeans(n_clusters=self.n_worker_group, n_init=10).fit_predict(B_flat)
            # Potentially do a 2D label_swap if you want stable group IDs.
    
            # -----------------------------------------------------------
            # E) Check Loss and Convergence
            # -----------------------------------------------------------
            loss_cur = self._mc_loss_func(data)
            loss_history.append(loss_cur)
    
            err = abs(loss_cur - loss_prev) / (abs(loss_prev) if loss_prev != 0 else 1)
            if verbose > 0:
                print(f"Iter: {iter_count}, loss: {loss_cur:.4f}, err: {err:.4e}")
    
            if err < epsilon:
                break
    
            loss_prev = loss_cur
    
        # Done
        self.iter = iter_count
        if verbose > 0:
            print("Fitting complete.")
        return self.B,self.V
