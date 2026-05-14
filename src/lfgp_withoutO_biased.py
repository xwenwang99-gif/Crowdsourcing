# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:19:38 2025

@author: wangl
"""
from src.dawid_skene_model import DawidSkeneModel
import numpy as np
import numpy_indexed as npi
from sklearn.cluster import KMeans
from itertools import permutations
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import mode
from collections import Counter
from scipy import stats
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class LFGP():
    def __init__(self, lf_dim=3, n_worker_group=2, lambda1=1, lambda2_0=1, lambda2_1=1):

        # Specify hyper-parameters

        self.lf_dim = lf_dim                    # dimension of latent factors     # number of worker subgroups
        self.lambda1 = lambda1                  # penalty coefficient for task subgrouping
        self.lambda2_0 = lambda2_0                  # penalty coefficient for worker subgrouping
        self.lambda2_1 = lambda2_1                  # penalty coefficient for worker subgrouping
        
        
    def to_torch(self, x, dtype=torch.float32):
        """Convert numpy array or tensor to float32 CUDA tensor."""
        if isinstance(x, torch.Tensor):
            return x.to(DEVICE, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=DEVICE)


    def to_numpy(self,x):
        return x.detach().cpu().numpy()
        
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
        worker = np.unique(data[:, 1])
        acc        = np.zeros((self.n_worker, self.n_task_group))
        bias_score = np.zeros((self.n_worker, self.n_task_group))
        member     = np.zeros((self.n_worker, self.n_task_group), dtype=int)
    
        for t_group in range(self.n_task_group):
            for i in range(self.n_worker):
                crowd_w = data[
                    (data[:, 1] == worker[i]) &
                    (label[data[:, 0].astype(int), 1] == t_group)
                ]
                if crowd_w.shape[0] == 0:
                    acc[i, t_group]        = 0
                    bias_score[i, t_group] = 0
                    continue
    
                task_w        = crowd_w[:, 0].astype(int)
                true_labels   = label[np.isin(label[:, 0], task_w), 1]
                worker_labels = crowd_w[:, 2]
    
                # Accuracy: fraction of correct labels
                acc[i, t_group] = np.mean(true_labels == worker_labels)
    
                # Bias score: among incorrect responses, how concentrated
                # are they on a single wrong label?
                # A purely biased worker scores 1.0, a random worker scores ~1/K
                wrong_mask = (true_labels != worker_labels)
                wrong_labels = worker_labels[wrong_mask]
                if len(wrong_labels) > 0:
                    counts = np.bincount(wrong_labels.astype(int),
                                         minlength=self.n_task_group)
                    bias_score[i, t_group] = counts.max() / len(wrong_labels)
                else:
                    # All correct — HQ worker, bias score irrelevant
                    bias_score[i, t_group] = 0
    
        for t_group in range(self.n_task_group):
            median_acc = np.median(acc[:, t_group])
    
            for i in range(self.n_worker):
                if acc[i, t_group] > median_acc:
                    # Above median accuracy → HQ
                    member[i, t_group] = 1
                else:
                    # Below median — use bias score to distinguish
                    # biased (concentrated errors) from LQ (random errors)
                    median_bias = np.median(
                        bias_score[acc[:, t_group] <= median_acc, t_group]
                    )
                    if bias_score[i, t_group] > median_bias:
                        member[i, t_group] = 2   # biased
                    else:
                        member[i, t_group] = 0   # LQ
    
        return member


    
    def _init_mc_params(self, data, task_lf, worker_lf, scheme):

        # initialize model parameters for multicategory crowdsourcing
        # two initialization schemes are available: mv and random

        if scheme == "mv":

            task_member = self._init_task_member_mv(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member
            '''         
            noise_scale = 0.1

            A = task_lf + np.random.normal(0, noise_scale, size=task_lf.shape)
            B = worker_lf + np.random.normal(0, noise_scale, size=worker_lf.shape)
            '''
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
        
    def _init_task_member_mv(self, data):

        # initialize model parameters (task initial subgroup membership) using majority voting scheme

        task = np.unique(data[:, 0]) # task index
        label = np.zeros((self.n_task, 2))
        label[:, 0] = task

        for i in range(self.n_task):
            val, _ = stats.mode(data[data[:, 0] == task[i], 2]) # get the majority of crowdsourced label for each task
            label[i, 1] = val                                # assign the majority voted label to the initial label

        return label
        
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
    
    def mc_loss_func_gpu(self,data_t, task_id, worker_id, A, B, U, V,
                     clusters, lambda1, lambda2_0, lambda2_1, lf_dim, n_task_group):
        """
        Fully vectorized loss — no Python loop over records.
        data_t   : (n_record, 3) torch tensor  [task, worker, label]
        clusters : (2, lf_dim, n_task_group) torch tensor
        """
        # ── likelihood ──
        A_obs = A[task_id]                      # (R, k)
        B_obs = B[worker_id]                    # (R, C, k)
        labels = data_t[:, 2].long()            # (R,)
    
        logits = torch.einsum('rk,rck->rc', A_obs, B_obs)   # (R, C)
        log_probs = torch.log_softmax(logits, dim=1)          # (R, C)
        loss = -log_probs[torch.arange(len(labels)), labels].sum()
    
        # ── penalty 1: task group ──
        penalty1 = torch.tensor(0.0, device=DEVICE)
        for g in torch.unique(U):
            mask = (U == g)
            centroid = A[mask].mean(0)
            penalty1 += lambda1 * torch.sum((A[mask] - centroid) ** 2)
    
        # ── penalty 2: worker group ──
        penalty2 = torch.tensor(0.0, device=DEVICE)
        for j in range(n_task_group):
            mask0 = (V[:, j] == 0)
            mask1 = (V[:, j] == 1)
            mask0 = (V[:, j] == 0)
            mask1 = (V[:, j] == 1)
            mask2 = (V[:, j] == 2)
            if mask0.any():
                penalty2 += lambda2_0 * torch.linalg.norm(B[mask0, j, :])
            if mask1.any():
                center1 = clusters[1, :, j]
                penalty2 += lambda2_1 * torch.linalg.norm(B[mask1, j, :] - center1)
            if mask2.any():
                center2 = clusters[2, :, j]
                penalty2 += lambda2_1 * torch.linalg.norm(B[mask2, j, :] - center2)
    
        return (loss + penalty1 + penalty2).item()
    
    def comp_centroid_gpu(self,A, B, U, V, n_task_group):
        """
        A : (n_task, k)
        B : (n_worker, C, k)
        U : (n_task,)   long
        V : (n_worker, C)  long
        """
        n_task, k = A.shape
        n_worker = B.shape[0]
    
        Centroid_A = torch.zeros_like(A)
        Centroid_B = torch.zeros_like(B)
    
        for g in range(n_task_group):
            mask = (U == g)
            if mask.any():
                Centroid_A[mask] = A[mask].mean(0)
    
        for t_group in range(n_task_group):
            v_col = V[:, t_group]
            for g in torch.unique(v_col):
                mask = (v_col == g)
                if mask.any():
                    Centroid_B[mask, t_group, :] = B[mask, t_group, :].mean(0)
    
        return Centroid_A, Centroid_B

    
    def multinomial_reg1_batched(self,A, B_all, Y_all, obs_idx_per_task,
                              lambd, Alpha, max_iter=10, lr=0.001, tol=1e-1):
        """
        Batched gradient descent for ALL tasks simultaneously.
    
        A        : (n_task, k)         - task latent factors
        B_all    : (n_worker, C, k)    - worker latent factors
        Y_all    : (n_record,)         - labels (int)
        obs_idx_per_task: list of (obs_worker_indices, obs_labels) per task
                          precomputed once before the loop
        Alpha    : (n_task, k)         - task group centroids
        """
        # We update each task independently but in vectorized form per task.
        # For truly batched updates across tasks, tasks must have the same number
        # of observations — which they generally don't. We therefore vectorize
        # WITHIN each task (eliminate the inner n/c loops) and call torch.vmap
        # or a simple per-task loop that is fast because all ops are tensor ops.
    
        n_task, k = A.shape
    
        for t in range(n_task):
            worker_idx, obs_labels = obs_idx_per_task[t]
            if len(worker_idx) == 0:
                continue
    
            B = B_all[worker_idx]          # (N, C, k)
            Y = obs_labels                 # (N,)
            beta = A[t].clone()            # (k,)
            centroid = Alpha[t]            # (k,)
    
            for _ in range(max_iter):
                # conc1 = B @ beta  →  (N, C)
                conc1 = (B @ beta)                          # (N, C)
                conc1 = torch.softmax(conc1, dim=1)         # (N, C)
    
                # grad = sum_n [ -B[n,Y[n],:] + sum_c softmax_nc * B[n,c,:] ]
                #       + 2*lambd*(beta - centroid)
                # B[n,Y[n],:] gathered:
                B_true = B[torch.arange(len(Y)), Y]         # (N, k)
                # weighted sum of B:  einsum nc,nck->k
                B_weighted = torch.einsum('nc,nck->k', conc1, B)  # (k,)
    
                # REPLACE with just this one line:
                grad = B_weighted - B_true.sum(0) + 2 * lambd * (beta - centroid)
    
                if torch.linalg.norm(grad) <= tol:
                    break
                beta = beta - lr * grad
    
            A[t] = beta
    
        return A

    
    def multinomial_reg2_batched(self,B, A_all, Y_all, obs_idx_per_worker_group,
                              V, lambda2_0, lambda2_1, clusters,
                              n_task_group, max_iter=10, lr=0.001, tol=1e-1):
        """
        Batched gradient descent for ALL (worker, group) pairs simultaneously.
    
        B        : (n_worker, C, k)
        A_all    : (n_task, k)
        obs_idx_per_worker_group: list of (task_indices, labels) per (w, group)
        """
        n_worker = B.shape[0]
    
        for w in range(n_worker):
            for group in range(n_task_group):
                task_idx, obs_labels = obs_idx_per_worker_group[w][group]
                if len(task_idx) == 0:
                    continue
    
                A = A_all[task_idx]             # (M, k)
                Y = obs_labels                  # (M,)
                beta = B[w].clone()             # (C, k)
                worker_group = int(V[w, group].item())
                if worker_group == 0:
                    lambd = lambda2_0
                    centroid = clusters[0, :, group]   # LQ: pulls toward origin
                elif worker_group == 1:
                    lambd = lambda2_1
                    centroid = clusters[1, :, group]   # HQ: pulls toward HQ centroid
                else:  # worker_group == 2
                    lambd = lambda2_1                  # biased: same strength as HQ
                    centroid = clusters[2, :, group]   # pulls toward biased centroid
    
                for _ in range(max_iter):
                    # conc1[m,c] = A[m,:] @ beta[c,:]  →  A @ beta.T  (M, C)
                    conc1 = torch.softmax(A @ beta.T, dim=1)  # (M, C)
    
                    # one-hot for true labels
                    one_hot = torch.zeros_like(conc1)
                    one_hot[torch.arange(len(Y)), Y] = 1.0
    
                    # grad[c,:] = sum_m A[m,:] * (conc1[m,c] - one_hot[m,c])
                    # = (conc1 - one_hot).T @ A   →  (C, k)
                    grad = (conc1 - one_hot).T @ A  # (C, k)
                    grad += 2 * lambd * (beta - centroid.unsqueeze(0))
    
                    if torch.linalg.norm(grad) <= tol:
                        break
                    beta = beta - lr * grad
    
                B[w] = beta
    
        return B
    
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
    
    def new_kmeans_gpu_3cluster(self, X, lf_dim, n_worker, max_iter=300, tol=1e-4):
        """
        3-cluster KMeans with the following structure matching the new penalty:
          center[0] = fixed at origin          (LQ workers,     W=0)
          center[1] = free HQ centroid         (HQ workers,     W=1)
          center[2] = biased centroid,         (biased workers, W=2)
                      orthogonalized against center[1] after each update
    
        X      : (n_worker, lf_dim) torch tensor on DEVICE
        Returns: labels  (n_worker,)     — 0=LQ, 1=HQ, 2=biased
                 centers (3, lf_dim)     — both on DEVICE
        """
        centers = torch.zeros(3, lf_dim, device=DEVICE)
    
        # Initialise center[1] (HQ) as the worker with the largest norm
        norms = torch.linalg.norm(X, dim=1)
        centers[1] = X[torch.argmax(norms)]
    
        # Initialise center[2] (biased) as the worker with the largest norm
        # in the direction orthogonal to center[1]
        proj = (X @ centers[1]) / (torch.linalg.norm(centers[1]) ** 2 + 1e-12)
        X_orth = X - proj.unsqueeze(1) * centers[1].unsqueeze(0)  # (n_worker, lf_dim)
        orth_norms = torch.linalg.norm(X_orth, dim=1)
        centers[2] = X[torch.argmax(orth_norms)]
        # Immediately orthogonalize center[2] against center[1]
        centers[2] = centers[2] - (
            (centers[2] @ centers[1]) /
            (torch.linalg.norm(centers[1]) ** 2 + 1e-12)
        ) * centers[1]
    
        labels = torch.zeros(n_worker, dtype=torch.long, device=DEVICE)
    
        for _ in range(max_iter):
            # ── Assignment step ──────────────────────────────────────────
            # distances: (n_worker, 3)
            diff = X.unsqueeze(1) - centers.unsqueeze(0)   # (n_worker, 3, lf_dim)
            distances = torch.linalg.norm(diff, dim=2)      # (n_worker, 3)
            new_labels = torch.argmin(distances, dim=1)
    
            # ── Update step ──────────────────────────────────────────────
            new_centers = centers.clone()
    
            # center[0] stays fixed at origin — never updated
            # center[1]: mean of HQ-assigned workers
            pts_hq = X[new_labels == 1]
            if len(pts_hq) > 0:
                new_centers[1] = pts_hq.mean(0)
    
            # center[2]: mean of biased-assigned workers, then project ⊥ center[1]
            pts_bias = X[new_labels == 2]
            if len(pts_bias) > 0:
                gamma_raw = pts_bias.mean(0)
                # Gram-Schmidt: remove component along center[1]
                new_centers[2] = gamma_raw - (
                    (gamma_raw @ new_centers[1]) /
                    (torch.linalg.norm(new_centers[1]) ** 2 + 1e-12)
                ) * new_centers[1]
    
            # ── Convergence check ────────────────────────────────────────
            if torch.all(torch.abs(new_centers - centers) < tol):
                labels = new_labels
                centers = new_centers
                break
    
            centers = new_centers
            labels = new_labels
    
        # ── Label disambiguation ─────────────────────────────────────────
        # Ensure center[1] (HQ) has larger norm than center[2] (biased).
        # If not, swap labels 1 and 2 so the stronger non-zero centroid
        # is always called HQ — the eigendecomposition step will
        # resolve which is truly HQ vs biased downstream.
        if torch.linalg.norm(centers[2]) > torch.linalg.norm(centers[1]):
            swap_mask = (labels == 1) | (labels == 2)
            labels[swap_mask] = 3 - labels[swap_mask]   # 1↔2
            centers[[1, 2]] = centers[[2, 1]]
    
        return labels, centers



  
    
    def _mc_fit(self, data, key, scheme="mv", maxiter=50, epsilon=1e-5, verbose=0, A = None, B = None):
        """
        GPU-accelerated drop-in replacement for _mc_fit.
        self must have: A, B, U, V, lf_dim, n_task, n_worker, n_task_group,
                        lambda1, lambda2_0, lambda2_1, n_record,
                        _init_mc_params, label_swap
        """
        acc_with_iter = []
        self._init_mc_params(data, A, B, scheme=scheme)
        
        self.U = self.U.astype(int)
        self.V = self.V.astype(int)
        new_U = self._mc_infer(data)
        acc_with_iter.append(np.mean(new_U == key))

    
        # ── Move everything to GPU ──
        A = self.to_torch(self.A)          # (n_task, k)
        B = self.to_torch(self.B)          # (n_worker, C, k)
        U = self.to_torch(self.U, dtype=torch.long)   # (n_task,)
        V = self.to_torch(self.V, dtype=torch.long)   # (n_worker, C)
    
        data_np = data
        data_t = self.to_torch(data)       # (n_record, 3)
    
        task_ids_np, task_idx_np = np.unique(data[:, 0], return_inverse=True)
        worker_ids_np, worker_idx_np = np.unique(data[:, 1], return_inverse=True)
    
        task_idx_t  = self.to_torch(task_idx_np,  dtype=torch.long)
        worker_idx_t = self.to_torch(worker_idx_np, dtype=torch.long)
    
        n_task_group = self.n_task_group
        lf_dim = self.lf_dim
    
        # ── Precompute observation indices (done once, on CPU for indexing) ──
        # obs_idx_per_task[t] = (worker_indices_tensor, labels_tensor)
        obs_idx_per_task = []
        for t in range(self.n_task):
            mask = (data_np[:, 0] == task_ids_np[t])
            w_idx = self.to_torch(worker_idx_np[mask], dtype=torch.long)
            labels = self.to_torch(data_np[mask, 2].astype(int), dtype=torch.long)
            obs_idx_per_task.append((w_idx, labels))
    
        # obs_idx_per_worker_group[w][g] = (task_indices_tensor, labels_tensor)
        obs_idx_per_worker_group = []
        for w in range(self.n_worker):
            worker_groups = []
            for group in range(n_task_group):
                mask = ((data_np[:, 1] == worker_ids_np[w]) &
                        (U[data_np[:, 0].astype(int)].cpu().numpy() == group))
                t_idx = self.to_torch(task_idx_np[mask], dtype=torch.long)
                labels = self.to_torch(data_np[mask, 2].astype(int), dtype=torch.long)
                worker_groups.append((t_idx, labels))
            obs_idx_per_worker_group.append(worker_groups)
    
        clusters = torch.zeros(3, lf_dim, n_task_group, device=DEVICE)
        loss_prev = float("inf")
        loss_history = []
    
        if verbose > 0:
            print(f"Starting GPU optimization on {DEVICE}...")
    
        V_cur = V.clone()
            
        for iter_count in range(maxiter):
            if verbose > 0:
                print(f"\nIteration {iter_count + 1}/{maxiter}")
    
            A_prev = A.clone()
            B_prev = B.clone()
            U_prev = U.clone()
            V_prev = V.clone()
    
            Alpha, _ = self.comp_centroid_gpu(A_prev, B_prev, U_prev, V_prev, n_task_group)
    
            # ── Update A (all tasks) ──
            A = self.multinomial_reg1_batched(
                A, B_prev, None, obs_idx_per_task,
                self.lambda1, Alpha
            )
    
            # ── Update B (all workers × groups) ──
            B = self.multinomial_reg2_batched(
                B, A, None, obs_idx_per_worker_group,
                V, self.lambda2_1, self.lambda2_0, clusters,
                n_task_group
            )
    
            # ── Update U via KMeans (sklearn on CPU — A is small) ──
            A_np = self.to_numpy(A)
            U_cur_np = KMeans(n_clusters=n_task_group, n_init=10).fit_predict(A_np)
            U_cur_np = self.label_swap(U_cur_np, self.to_numpy(U_prev))
            U = self.to_torch(U_cur_np, dtype=torch.long)
    
            # ── Update V via GPU KMeans ──
            for t in range(n_task_group):
                B_slice = B[:, t, :]   # (n_worker, k)
                labels, centers = self.new_kmeans_gpu_3cluster(B_slice, lf_dim, self.n_worker)
                V_cur[:, t] = labels
                clusters[:, :, t] = centers
    
            V = V_cur.clone()
    
            # ── Recompute obs indices for workers (U changed) ──
            obs_idx_per_worker_group = []
            U_np = self.to_numpy(U).astype(int)
            for w in range(self.n_worker):
                worker_groups = []
                for group in range(n_task_group):
                    mask = ((data_np[:, 1] == worker_ids_np[w]) &
                            (U_np[data_np[:, 0].astype(int)] == group))
                    t_idx = self.to_torch(task_idx_np[mask], dtype=torch.long)
                    labels_t = self.to_torch(data_np[mask, 2].astype(int), dtype=torch.long)
                    worker_groups.append((t_idx, labels_t))
                obs_idx_per_worker_group.append(worker_groups)
    
            # ── Loss ──
            loss_cur = self.mc_loss_func_gpu(
                data_t, task_idx_t, worker_idx_t,
                A, B, U, V, clusters,
                self.lambda1, self.lambda2_0, self.lambda2_1,
                lf_dim, n_task_group
            )
            loss_history.append(loss_cur)
    
            if verbose > 0:
                change = abs(loss_prev - loss_cur) / abs(loss_prev) if loss_prev != float("inf") else float("inf")
                print(f"Loss: {loss_cur:.6f}, Change: {change:.6e}")
    
            if abs(loss_prev - loss_cur) / abs(loss_prev) < epsilon:
               if verbose > 0:
                   print("Convergence achieved.")
               self.U = self.to_numpy(U).astype(int)
               self.V = self.to_numpy(V).astype(int)
               new_U = self._mc_infer(data)
               acc_with_iter.append(np.mean(new_U == key))
               break
    
            loss_prev = loss_cur
            
            self.U = self.to_numpy(U).astype(int)
            self.V = self.to_numpy(V).astype(int)
            new_U = self._mc_infer(data)
            
            #Find the clustering accuracy after each iteration
            acc_with_iter.append(np.mean(new_U == key))            
    
        if verbose > 0:
            print("Optimization complete.")
    
        # ── Move results back to CPU/numpy to match original API ──
        self.A = self.to_numpy(A)
        self.B = self.to_numpy(B)
        self.U = self.to_numpy(U)
        self.V = self.to_numpy(V)
        clusters_np = self.to_numpy(clusters)
        
        plt.plot(range(len(acc_with_iter)), acc_with_iter)
        plt.show()
    
        return self.A, self.B, self.U, self.V, clusters_np
    
    
    
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
    
    def task_acc(self, data, key):
        new_U = self.label_swap(self.U, key)
        return np.mean(new_U == key)
            
            
            
            
                
            
                
            