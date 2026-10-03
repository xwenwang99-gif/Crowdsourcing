# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:19:38 2025

@author: wangl
"""
from src.dawid_skene_model import DawidSkeneModel
import numpy as np
import numpy_indexed as npi
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import mode
from collections import Counter
from scipy import stats
from scipy.optimize import linear_sum_assignment
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
        """Initialize worker tiers as 0=LQ and 1=HQ using within-group accuracy."""
        worker = np.unique(data[:, 1])
        acc = np.zeros((self.n_worker, self.n_task_group))
        member = np.zeros((self.n_worker, self.n_task_group), dtype=int)

        for t_group in range(self.n_task_group):
            for i in range(self.n_worker):
                crowd_w = data[
                    (data[:, 1] == worker[i]) &
                    (label[data[:, 0].astype(int), 1] == t_group)
                ]

                if crowd_w.shape[0] == 0:
                    acc[i, t_group] = 0
                    continue

                task_w = crowd_w[:, 0].astype(int)
                reference_labels = label[np.isin(label[:, 0], task_w), 1]
                worker_labels = crowd_w[:, 2]

                acc[i, t_group] = np.mean(reference_labels == worker_labels)

        for t_group in range(self.n_task_group):
            median_acc = np.median(acc[:, t_group])
            member[:, t_group] = (acc[:, t_group] > median_acc).astype(int)

        return member

    def _init_mc_params(self, data, task_lf, worker_lf, scheme, U_init=None, V_init=None, clusters_init=None):

        # initialize model parameters for multicategory crowdsourcing
        # two initialization schemes are available: mv and random

        if scheme == "mv":

            task_member = self._init_task_member_mv(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member
 
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
            
        elif scheme == "warm":
            required = {
                "A_init": task_lf,
                "B_init": worker_lf,
                "U_init": U_init,
                "V_init": V_init,
                "clusters_init": clusters_init,
            }
        
            missing = [
                name for name, value in required.items()
                if value is None
            ]
        
            if missing:
                raise ValueError(
                    "scheme='warm' requires: "
                    + ", ".join(missing)
                )
            U = np.asarray(U_init).astype(int).copy()
            V = np.asarray(V_init).astype(int).copy()

            if not np.all(np.isin(V, [0, 1])):
                raise ValueError(
                    "HQ/LQ model requires warm-start V_init to contain only 0=LQ and 1=HQ."
                )

            task_member = np.column_stack([np.arange(self.n_task), U]).astype(float)
            A = task_lf.copy()
            A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
            B = worker_lf.copy()
            
        elif scheme == "task_oracle":
            if U_init is None:
                raise ValueError("scheme='task_oracle' requires U_init.")
            U = np.asarray(U_init).astype(int).copy()
            task_member = np.column_stack([np.arange(self.n_task), U]).astype(float)
            
            V = self._init_worker_member_acc(data, task_member)
  
            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(V)
            
        elif scheme == "worker_oracle":
            if V_init is None:
                raise ValueError(
                    "scheme='worker_oracle' requires V_init."
                )
        
            # Task grouping is NOT oracle.
            # Start tasks exactly as the ordinary likelihood fit does.
            
            V_init = (
                V_init == 1
            ).astype(int)
            task_member = self._init_task_member_ds(data)
        
            U = task_member[:, 1].astype(int)
        
            # Worker tiers ARE oracle.
            V = np.asarray(V_init).astype(int).copy()
        
            if V.shape != (self.n_worker, self.n_task_group):
                raise ValueError(
                    f"V_init has shape {V.shape}; expected "
                    f"{(self.n_worker, self.n_task_group)}."
                )
        
            # Initialize latent factors using estimated U but oracle V.
            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(V)

            
                      
        U = U.astype(int)
        V = V.astype(int)

        if not np.all(np.isin(V, [0, 1])):
            raise ValueError(
                "HQ/LQ model requires worker memberships in {0, 1}."
            )

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
        
        lf /= np.linalg.norm(lf, axis=1, keepdims=True) + 1e-12
        return lf
    


    def _init_worker_lf_gp(self, member, hq_scale=2.0):
        """
        Initialize worker latent factors for a two-tier model:
            0 = LQ, centered at the origin
            1 = HQ, centered at a nonzero group-specific direction
        """
        member = np.asarray(member, dtype=int)
        if not np.all(np.isin(member, [0, 1])):
            raise ValueError("HQ/LQ model requires worker memberships in {0, 1}.")

        lf = np.zeros((self.n_worker, self.n_task_group, self.lf_dim))

        for t_group in range(self.n_task_group):
            direction = np.random.randn(self.lf_dim)
            direction /= np.linalg.norm(direction) + 1e-12
            hq_centroid = hq_scale * direction

            for i in range(self.n_worker):
                centroid = (
                    np.zeros(self.lf_dim)
                    if member[i, t_group] == 0
                    else hq_centroid
                )
                lf[i, t_group, :] = np.random.multivariate_normal(
                    centroid,
                    0.2 * np.eye(self.lf_dim),
                )

        return lf

    def worker_centers_from_V(self, B, V, n_task_group, worker_active_t=None):
        """
        Construct two worker centers per task group:
            center 0 = fixed LQ center at the origin
            center 1 = mean HQ latent factor
        """
        if worker_active_t is None:
            worker_active_t = torch.ones(
                B.shape[0],
                dtype=torch.bool,
                device=B.device,
            )

        clusters = torch.zeros(
            2,
            self.lf_dim,
            n_task_group,
            device=B.device,
            dtype=B.dtype,
        )

        for g in range(n_task_group):
            hq_mask = (V[:, g] == 1) & worker_active_t

            # LQ center remains fixed at the origin.
            clusters[0, :, g] = 0.0

            if hq_mask.any():
                clusters[1, :, g] = B[hq_mask, g, :].mean(0)

        return clusters


    def mc_loss_func_gpu(
        self,
        data_t,
        task_id,
        worker_id,
        A,
        B,
        U,
        V,
        clusters,
        lambda1,
        lambda2_0,
        lambda2_1,
        lf_dim,
        n_task_group,
        worker_active_t=None,
    ):
        if worker_active_t is None:
            worker_active_t = torch.ones(
                B.shape[0],
                dtype=torch.bool,
                device=B.device,
            )

        labels = data_t[:, 2].long()
        A_obs = A[task_id]

        # ==================================================
        # Multinomial label likelihood
        # ==================================================
        B_obs = B[worker_id]                         # (R, C, k)

        logits = torch.einsum(
            "rk,rck->rc",
            A_obs,
            B_obs,
        )

        log_probs = torch.log_softmax(
            logits,
            dim=1,
        )

        loss = -log_probs[
            torch.arange(
                len(labels),
                device=DEVICE,
            ),
            labels,
        ].sum()

        # ==================================================
        # Penalty 1: task grouping
        # ==================================================
        penalty1 = torch.tensor(
            0.0,
            device=DEVICE,
        )

        for g in torch.unique(U):
            mask = U == g
            centroid = A[mask].mean(0)

            penalty1 += (
                lambda1
                * torch.sum(
                    (A[mask] - centroid) ** 2
                )
            )

        # ==================================================
        # Penalty 2: worker grouping (0=LQ, 1=HQ)
        # ==================================================
        penalty2 = torch.tensor(
            0.0,
            device=DEVICE,
        )

        for g in range(n_task_group):
            mask0 = (V[:, g] == 0) & worker_active_t
            mask1 = (V[:, g] == 1) & worker_active_t

            if mask0.any():
                penalty2 += (
                    lambda2_0
                    * torch.sum(
                        (B[mask0, g, :] - clusters[0, :, g]) ** 2
                    )
                )

            if mask1.any():
                penalty2 += (
                    lambda2_1
                    * torch.sum(
                        (B[mask1, g, :] - clusters[1, :, g]) ** 2
                    )
                )

        return (
            loss
            + penalty1
            + penalty2
        ).item()

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

    
    def multinomial_reg1_batched(
        self,
        A,
        B_all,
        obs_idx_per_task,
        lambd,
        Alpha,
        max_iter=10,
        lr=0.001,
        tol=1e-1,
    ):
        """Update task latent factors A under the multinomial label likelihood."""
    
        n_task, k = A.shape
    
        for t in range(n_task):
    
            worker_idx, obs_labels = obs_idx_per_task[t]
    
            if len(worker_idx) == 0:
                continue
    
            Y = obs_labels.long()
            beta = A[t].clone()
            centroid = Alpha[t]
    
            B = B_all[worker_idx]       # (N, C, k)
    
            for _ in range(max_iter):
    
                logits = B @ beta
                prob = torch.softmax(
                    logits,
                    dim=1,
                )
    
                B_true = B[
                    torch.arange(
                        len(Y),
                        device=Y.device,
                    ),
                    Y,
                ]
    
                B_weighted = torch.einsum(
                    "nc,nck->k",
                    prob,
                    B,
                )
    
                grad = (
                    B_weighted
                    - B_true.sum(0)
                    + 2 * lambd
                    * (beta - centroid)
                )
    
                if torch.linalg.norm(grad) <= tol:
                    break
    
                beta = beta - lr * grad
                beta = beta / (torch.linalg.norm(beta) + 1e-12)
    
            A[t] = beta
    
        return A

    

    def multinomial_reg2_batched(
        self,
        B,
        A_all,
        obs_idx_per_worker,
        V,
        lambda2_0,
        lambda2_1,
        clusters,
        n_task_group,
        max_iter=10,
        lr=0.001,
        tol=1e-1,
    ):
        """
        Update worker latent factors B under the multinomial label likelihood.

        Worker tiers are binary:
            0 = LQ
            1 = HQ
        """
        n_worker = B.shape[0]
        n_classes = B.shape[1]

        if n_classes != n_task_group:
            raise ValueError(
                "Current implementation assumes n_classes == n_task_group."
            )

        for w in range(n_worker):
            task_idx, obs_labels = obs_idx_per_worker[w]

            if len(task_idx) == 0:
                continue

            A = A_all[task_idx]          # (M_w, k)
            Y = obs_labels.long()        # (M_w,)

            # Update all class-specific worker factors together.
            beta = B[w].clone()          # (C, k)

            for _ in range(max_iter):
                logits = A @ beta.T      # (M_w, C)
                prob = torch.softmax(
                    logits,
                    dim=1,
                )

                one_hot = torch.zeros_like(prob)
                one_hot.scatter_(
                    1,
                    Y.unsqueeze(1),
                    1.0,
                )

                grad = (
                    (prob - one_hot).T @ A
                )                        # (C, k)

                # Add the two-tier worker penalty gradient.
                for c in range(n_classes):
                    worker_tier = int(V[w, c].item())

                    if worker_tier == 0:
                        lambd = lambda2_0
                        centroid = clusters[0, :, c]
                    elif worker_tier == 1:
                        lambd = lambda2_1
                        centroid = clusters[1, :, c]
                    else:
                        raise ValueError(
                            f"HQ/LQ model requires worker tier 0 or 1, got {worker_tier}."
                        )

                    grad[c] += (
                        2 * lambd
                        * (beta[c] - centroid)
                    )

                if torch.linalg.norm(grad) <= tol:
                    break

                beta = beta - lr * grad

            B[w] = beta

        return B

    def label_swap(self, Grp_cur, Grp_prev):
        Grp_cur = np.asarray(Grp_cur, dtype=int)
        Grp_prev = np.asarray(Grp_prev, dtype=int)

        if Grp_cur.shape != Grp_prev.shape:
            raise ValueError("Grp_cur and Grp_prev must have the same shape.")

        n_groups = max(Grp_cur.max(), Grp_prev.max()) + 1

        counts = np.zeros((n_groups, n_groups), dtype=np.int64)
        np.add.at(counts, (Grp_cur, Grp_prev), 1)

        row_ind, col_ind = linear_sum_assignment(-counts)

        mapping = np.arange(n_groups)
        mapping[row_ind] = col_ind

        return mapping[Grp_cur]
        

    def new_kmeans_gpu_2cluster(
        self,
        X,
        lf_dim,
        n_worker,
        max_iter=300,
        tol=1e-4,
        clusters_init=None,
    ):
        """
        Two-cluster KMeans for worker factors:
            cluster 0 = LQ, fixed at the origin
            cluster 1 = HQ, freely estimated
        """
        if clusters_init is not None:
            centers = clusters_init.clone().to(
                device=DEVICE,
                dtype=X.dtype,
            )
            if tuple(centers.shape) != (2, lf_dim):
                raise ValueError(
                    f"clusters_init has shape {tuple(centers.shape)}; "
                    f"expected {(2, lf_dim)}."
                )
            centers[0] = 0.0
        else:
            centers = torch.zeros(
                2,
                lf_dim,
                device=DEVICE,
                dtype=X.dtype,
            )
            norms = torch.linalg.norm(X, dim=1)
            centers[1] = X[torch.argmax(norms)]

        labels = torch.zeros(
            n_worker,
            dtype=torch.long,
            device=DEVICE,
        )

        for _ in range(max_iter):
            diff = X.unsqueeze(1) - centers.unsqueeze(0)
            new_labels = torch.argmin(
                torch.linalg.norm(diff, dim=2),
                dim=1,
            )

            new_centers = centers.clone()
            new_centers[0] = 0.0

            hq_points = X[new_labels == 1]
            if len(hq_points) > 0:
                new_centers[1] = hq_points.mean(0)

            converged = bool(
                torch.all(
                    torch.abs(new_centers - centers) < tol
                )
            )

            centers, labels = new_centers, new_labels

            if converged:
                break

        return labels, centers

    def _mc_fit(self, data, key, scheme="mv", maxiter=50, epsilon=1e-5, verbose=0,
                A_init=None, B_init=None, U_init=None, V_init=None,
                clusters_init=None, worker_active_mask=None):
        """
        Fit the latent-factor crowdsourcing model with two worker tiers:
            0 = LQ
            1 = HQ

        The LQ worker center is fixed at the origin; the HQ center is estimated
        separately within each task group.
        """
        acc_with_iter = []
        self._init_mc_params(data, A_init, B_init, scheme=scheme, U_init=U_init, V_init=V_init, clusters_init=clusters_init)
        
        if worker_active_mask is None:
            worker_active_mask = np.ones(self.n_worker, dtype=bool)
        else:
            worker_active_mask = np.asarray(worker_active_mask, dtype=bool)
        
            if worker_active_mask.shape != (self.n_worker,):
                raise ValueError(
                    f"worker_active_mask has shape {worker_active_mask.shape}; "
                    f"expected {(self.n_worker,)}."
                )

        # Removed workers are permanently treated as LQ during this fit.
        self.V[~worker_active_mask, :] = 0
        
        # The LQ center is fixed at zero, so removed workers stay at the origin.
        self.B[~worker_active_mask, :, :] = 0.0
        
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

        task_ids_np, task_idx_np = np.unique(data_np[:, 0], return_inverse=True)
        worker_ids_np, worker_idx_np = np.unique(data_np[:, 1], return_inverse=True)
        
        record_active = worker_active_mask[worker_idx_np]
        
        # Loss is evaluated only on retained workers.
        data_t = self.to_torch(data_np[record_active])
        task_idx_t = self.to_torch(task_idx_np[record_active], dtype=torch.long)
        worker_idx_t = self.to_torch(worker_idx_np[record_active], dtype=torch.long)
    
        n_task_group = self.n_task_group
        lf_dim = self.lf_dim
        
        worker_active_t = self.to_torch(worker_active_mask, dtype=torch.bool)
        active_idx = torch.where(worker_active_t)[0]
    
        # ── Precompute observation indices (done once, on CPU for indexing) ──
        # obs_idx_per_task[t] = (worker_indices_tensor, labels_tensor)
        obs_idx_per_task = []
        for t in range(self.n_task):
            mask = (data_np[:, 0] == task_ids_np[t]) & record_active
            w_idx = self.to_torch(worker_idx_np[mask], dtype=torch.long)
            labels = self.to_torch(data_np[mask, 2].astype(int), dtype=torch.long)
            obs_idx_per_task.append((w_idx, labels))
    
        obs_idx_per_worker = []
        # obs_idx_per_worker[w] = all observations made by worker w
        for w in range(self.n_worker):
            mask = ((worker_idx_np == w)& record_active)
            t_idx = self.to_torch(task_idx_np[mask],dtype=torch.long,)
            labels = self.to_torch(data_np[mask, 2].astype(int),dtype=torch.long,)
            obs_idx_per_worker.append((t_idx, labels))
        clusters = self.worker_centers_from_V(
            B,
            V,
            n_task_group,
            worker_active_t=worker_active_t,
        )
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
            
            #lambda1 = min(0.1*iter_count, self.lambda1)
            #lambda2_1 = min(0.1*iter_count, self.lambda2_1)
            #lambda2_0 = min(0.1*iter_count, self.lambda2_0)
            
            lambda1 = self.lambda1
            lambda2_1 = self.lambda2_1
            lambda2_0 = self.lambda2_0
    
            # ── Update A (all tasks) ──
            A = self.multinomial_reg1_batched(
                A,
                B_prev,
                obs_idx_per_task,
                lambda1,
                Alpha,
            )
                
            # ── Update B (all workers × groups) ──
            B = self.multinomial_reg2_batched(
                B,
                A,
                obs_idx_per_worker,
                V,
                lambda2_0,
                lambda2_1,
                clusters,
                n_task_group,
            )
    
            # ── Update U via KMeans (sklearn on CPU — A is small) ──
            # ── Update U via KMeans (skipped when the grouping is clamped) ──
            if scheme!="task_oracle":
                A_np = self.to_numpy(A)
                U_cur_np = KMeans(n_clusters=n_task_group, n_init=10, random_state=999).fit_predict(A_np)
                U_cur_np = self.label_swap(U_cur_np, self.to_numpy(U_prev).astype(int))
                U = self.to_torch(U_cur_np, dtype=torch.long)

    
            # ── Update V via GPU KMeans ──
            if scheme == "worker_oracle":
                clusters = self.worker_centers_from_V(
                    B,
                    V,
                    n_task_group,
                    worker_active_t=worker_active_t,
                )
            else:

                # Normal estimated-worker case
               for t in range(n_task_group):
                    # Cluster only workers retained after the spectral screen.
                    B_slice = B[active_idx, t, :]
                
                    if len(active_idx) < 2:
                        V_cur[:, t] = 0
                        continue
                
                    labels, centers = self.new_kmeans_gpu_2cluster(
                        B_slice,
                        lf_dim,
                        len(active_idx),
                        clusters_init=clusters[:, :, t],
                    )
                
                    # Removed workers stay LQ permanently.
                    V_cur[:, t] = 0
                    V_cur[active_idx, t] = labels                
                    clusters[:, :, t] = centers
                    
               V = V_cur.clone()
    
            # ── Loss ──
            loss_cur = self.mc_loss_func_gpu(
                data_t,
                task_idx_t,
                worker_idx_t,
                A,
                B,
                U,
                V,
                clusters,
                lambda1,
                lambda2_0,
                lambda2_1,
                lf_dim,
                n_task_group,
                worker_active_t=worker_active_t,
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
            
            #Find the clustering accuracy after each iteration
            acc_with_iter.append(self.task_acc(self.U, key))            
    
        if verbose > 0:
            print("Optimization complete.")
    
        # ── Move results back to CPU/numpy to match original API ──
        self.A = self.to_numpy(A)
        self.B = self.to_numpy(B)
        self.U = self.to_numpy(U)
        self.V = self.to_numpy(V)
        clusters_np = self.to_numpy(clusters)
        
        self.loss_history = list(loss_history)
        self.acc_history = list(acc_with_iter)
        
        print("KMeans cluster acc:",self.task_acc(U, key))
        print("Oracle-centroid acc:",self.oracle_centroid_task_acc(self.A,key,self.n_task_group))         
        return self.A, self.B, self.U, self.V, clusters_np
    

    def calculate_worker_accuracy(self, worker_label):
        """
        Evaluate HQ/LQ recovery.

        Ground-truth labels are binarized as HQ=1 and non-HQ=0, so a simulation
        may contain additional worker types without adding those types to the model.
        """
        worker_label = np.asarray(worker_label)

        if worker_label.ndim == 3:
            worker_label = np.argmax(worker_label, axis=2)

        worker_label = (worker_label == 1).astype(int)

        worker_accuracy = np.zeros((self.n_task_group, 4))

        for t in range(self.n_task_group):
            worker_accuracy[t, 0] = np.mean(
                self.V[:, t] == worker_label[:, t]
            )

            FP = np.sum(
                (worker_label[:, t] == 0)
                & (self.V[:, t] == 1)
            )
            TN = np.sum(
                (worker_label[:, t] == 0)
                & (self.V[:, t] == 0)
            )
            TP = np.sum(
                (worker_label[:, t] == 1)
                & (self.V[:, t] == 1)
            )
            FN = np.sum(
                (worker_label[:, t] == 1)
                & (self.V[:, t] == 0)
            )

            worker_accuracy[t, 1] = FP / (FP + TN) if (FP + TN) > 0 else np.nan
            worker_accuracy[t, 2] = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            worker_accuracy[t, 3] = TP / (TP + FP) if (TP + FP) > 0 else np.nan

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
    
    def oracle_centroid_task_acc(self, A, y_true, n_groups):
        A = np.asarray(A)
        y_true = np.asarray(y_true).astype(int)
    
        centers = np.zeros((n_groups, A.shape[1]))
    
        for g in range(n_groups):
            centers[g] = A[y_true == g].mean(axis=0)
    
        dist = np.sum(
            (A[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
    
        pred = np.argmin(dist, axis=1)
    
        return np.mean(pred == y_true)
    
    def task_acc(self, data, key):
        membership = self.label_swap(data, key)
        data = np.asarray(data, dtype=int)
        key = np.asarray(key, dtype=int)
        return np.mean(membership == key)
    

    def calculate_likelihood_worker_scores(
        self,
        clusters,
        temperature=1.0,
        priors=None,
        standardize=True,
        eps=1e-12,
    ):
        """
        Convert learned worker embeddings into soft HQ/LQ scores.

        Tier convention:
            0 = LQ
            1 = HQ
        """
        B = np.asarray(self.B, dtype=float)
        clusters = np.asarray(clusters, dtype=float)

        if B.ndim != 3:
            raise ValueError(
                "self.B must have shape (n_worker, n_task_group, lf_dim)."
            )

        n_worker, n_task_group, lf_dim = B.shape

        expected_shape = (2, lf_dim, n_task_group)
        if clusters.shape != expected_shape:
            raise ValueError(
                f"clusters must have shape {expected_shape}; got {clusters.shape}."
            )

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        # centers[g, tier, :] with tier 0=LQ and tier 1=HQ
        centers = np.transpose(clusters, (2, 0, 1))

        # dist_sq[j, g, tier]
        diff = B[:, :, None, :] - centers[None, :, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)

        # Larger evidence means closer to the corresponding tier center.
        log_evidence = -dist_sq / temperature

        if priors is not None:
            priors = np.asarray(priors, dtype=float)

            if priors.shape == (2,):
                log_prior = np.log(priors + eps)[None, None, :]
            elif priors.shape == (n_task_group, 2):
                log_prior = np.log(priors + eps)[None, :, :]
            else:
                raise ValueError(
                    "priors must have shape (2,) or (n_task_group, 2)."
                )

            log_evidence = log_evidence + log_prior

        max_log = np.max(log_evidence, axis=2, keepdims=True)
        exp_log = np.exp(log_evidence - max_log)
        tier_prob = exp_log / (
            np.sum(exp_log, axis=2, keepdims=True) + eps
        )

        # Probability of HQ under the two-center distance model.
        score = tier_prob[:, :, 1]

        # Positive values favor HQ over LQ.
        hq_advantage = (
            log_evidence[:, :, 1]
            - log_evidence[:, :, 0]
        )

        if standardize:
            mean_g = np.nanmean(score, axis=0, keepdims=True)
            std_g = np.nanstd(score, axis=0, keepdims=True)
            score_z = (score - mean_g) / (std_g + eps)
        else:
            score_z = None

        return {
            "score": score,
            "score_z": score_z,
            "tier_prob": tier_prob,
            "log_evidence": log_evidence,
            "dist_sq": dist_sq,
            "hq_advantage": hq_advantage,
        }
