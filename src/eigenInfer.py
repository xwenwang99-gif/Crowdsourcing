# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 18:02:19 2026

@author: wangl
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import seaborn as sns

def _hq_and_label_infer(pred_group, 
                        R_obs,
                        label,
                        worker_label,
                        n_task, 
                        n_worker,
                        n_task_groups,
                        n_worker_groups,
                        USE_TOP2_EIGEN,
                        LABEL_MODE = 'task',
                        verbose = False
                        ):
    hq_workers_pred = [None] * n_task_groups
    group_label_pred = np.full(n_task_groups, -1, dtype=int)
    task_label_pred = np.full(n_task, -1, dtype=int) 
    
    #####################################################
    # 1. For each predicted group:
    #    agreement → eigen → HQ workers → group label
    #####################################################
    for g in range(n_task_groups):
    
        # tasks predicted to be in group g
        tasks_g = np.where(pred_group == g)[0]
        if len(tasks_g) == 0:
            print(f"Pred group {g}: no tasks assigned, skipping")
            continue
    
        # labels of those tasks by all workers
        R_g = R_obs[tasks_g, :]     # (#tasks_in_group_pred, n_worker)
    
        # --- worker–worker agreement matrix on predicted group g ---
        agreement_g = np.zeros((n_worker, n_worker))
        for w1 in range(n_worker):
            for w2 in range(n_worker):
                valid = ~np.isnan(R_g[:, w1]) & ~np.isnan(R_g[:, w2])
                if valid.sum() == 0:
                    agreement_g[w1, w2] = 0
                else:
                    agreement_g[w1, w2] = np.mean(R_g[valid, w1] == R_g[valid, w2])
    
        # --- eigendecomposition ---
        if USE_TOP2_EIGEN:
            
            eigvals, eigvecs = np.linalg.eigh(agreement_g)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # top-2 eigenvectors
            V2 = eigvecs[:, :2]  # shape: (n_worker, 2)
            
            # optional: normalize each eigenvector (usually unnecessary)
            V2 = V2 / (np.linalg.norm(V2, axis=0, keepdims=True) + 1e-12)
            
            scores = np.linalg.norm(V2, axis=1)  # shape: (n_worker,)
            
            m = n_worker // n_worker_groups
            hq_workers_g = np.argsort(scores)[::-1][:m]
            hq_workers_pred[g] = hq_workers_g
            
        else:
            
            eigvals, eigvecs = np.linalg.eigh(agreement_g)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
        
            leading = eigvecs[:, 0]
            if leading.mean() < 0:
                leading = -leading
            leading = leading / (np.linalg.norm(leading) + 1e-12)
        
            # --- predicted HQ workers in this predicted group ---
            m = n_worker // n_worker_groups        # expected HQ count per true group
            hq_workers_g = np.argsort(np.abs(leading))[::-1][:m]
            hq_workers_pred[g] = hq_workers_g
            
            abs_vals = np.abs(leading)
        
    
        plt.figure(figsize=(6, 4))
        plt.hist(scores, bins=40, density=True, alpha=0.8, edgecolor="black")
        plt.axvline(np.partition(scores, -m)[-m], color="red", linestyle="--",
                    label=f"Top-{m} cutoff")
        plt.title(f"Task Group {g}: |Leading Eigenvector|_2 Distribution")
        plt.xlabel("|v_i|")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
        if verbose:
            print(f"Pred group {g}: {len(tasks_g)} tasks → {m} HQ workers predicted")
        if LABEL_MODE == 'group':
        
            # --- HQ majority voting for the *label* of this predicted group ---
            collected = []
        
            for t in tasks_g:
                labels_t = R_obs[t, hq_workers_g]
            
                # drop NaNs if they exist
                labels_t = labels_t[~np.isnan(labels_t)]
            
                if labels_t.size > 0:
                    collected.append(labels_t)
            
            # if nothing to vote on, skip safely
            if len(collected) == 0:
                if verbose:
                    print(f"Pred group {g}: no valid labels for voting, skipping")
                continue
            
            all_labels = np.concatenate(collected)
            
            mv = mode(all_labels, axis=None).mode
            if np.isnan(mv):
                if verbose:
                    print(f"Pred group {g}: mode is NaN, skipping")
                continue
            
            mv_label = int(mv)
            group_label_pred[g] = mv_label
            
            if verbose:        
                print(f"Pred group {g}: HQ-voted label = {mv_label}")
            
            task_label_pred = np.zeros(n_task, dtype=int)
            for g in range(n_task_groups):
                tasks_g = np.where(pred_group == g)[0]
                if len(tasks_g) == 0:
                    continue
                if group_label_pred[g] == -1:
                    # If no label estimated for this predicted group, default to 0 (or skip)
                    task_label_pred[tasks_g] = 0
                else:
                    task_label_pred[tasks_g] = group_label_pred[g]
        
        

        
        if LABEL_MODE == 'task':
        # ============================================
        # Task-wise HQ majority vote (instead of group-wise)
        # ============================================
        
         # final predicted label per task
        
        
            tasks_g = np.where(pred_group == g)[0]
            if len(tasks_g) == 0:
                print(f"Pred group {g}: no tasks assigned, skipping")
                continue
        
        
            for t in tasks_g:
                labels_t = R_obs[t, hq_workers_g]
        
                # drop NaNs
                labels_t = labels_t[~np.isnan(labels_t)]
        
                if labels_t.size == 0:
                    # no HQ labels observed for this task; leave as -1 or choose a fallback
                    continue
        
                mv = mode(labels_t, axis=None).mode
                if np.isnan(mv):
                    continue
        
                task_label_pred[t] = int(mv)
                
    task_accuracy = np.mean(task_label_pred == label)
    

    
    
    
    #########################################
    # 3. HQ worker identification accuracy
    #########################################
    # worker_label: shape (n_worker, n_groups), 
    # worker_label[w, h] = 1 if worker w is HQ for true group h.
    
    # Build predicted HQ sets for *true* groups:
    # For each true group h, gather HQ workers from all predicted groups g
    # that voted label = h.
    if verbose:
        pred_hq_by_true_group = [set() for _ in range(n_task_groups)]
        
        for g in range(n_task_groups):
            voted_label = group_label_pred[g]  # true group index 0..4, or -1
            if voted_label == -1:
                continue
            hq_g = hq_workers_pred[g]
            if hq_g is None or len(hq_g) == 0:
                continue
            pred_hq_by_true_group[voted_label].update(hq_g)
        
        print("\n==== HQ WORKER IDENTIFICATION (after estimated U + eigen + MV) ====")
        
        all_true_hq = set()
        all_pred_hq = set()
        
        for h in range(n_task_groups):
            true_hq_set = set(np.where(worker_label[:, h] == 1)[0])   # true HQ workers for true group h
            pred_hq_set = pred_hq_by_true_group[h]                    # predicted HQ workers for true group h
        
            all_true_hq |= true_hq_set
            all_pred_hq |= pred_hq_set
        
            tp = len(true_hq_set & pred_hq_set)
            fp = len(pred_hq_set - true_hq_set)
            fn = len(true_hq_set - pred_hq_set)
        
            precision = tp / (tp + fp + 1e-12)
            recall    = tp / (tp + fn + 1e-12)
            f1        = 2 * precision * recall / (precision + recall + 1e-12)
        
            print(f"\nTrue group {h}:")
            print(f"  True HQ count = {len(true_hq_set)}")
            print(f"  Pred HQ count = {len(pred_hq_set)}")
            print(f"  TP={tp}, FP={fp}, FN={fn}")
            print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Overall pooled HQ metrics
        tp_all = len(all_true_hq & all_pred_hq)
        fp_all = len(all_pred_hq - all_true_hq)
        fn_all = len(all_true_hq - all_pred_hq)
        
        precision_all = tp_all / (tp_all + fp_all + 1e-12)
        recall_all    = tp_all / (tp_all + fn_all + 1e-12)
        f1_all        = 2 * precision_all * recall_all / (precision_all + recall_all + 1e-12)
        
        print("\n==== OVERALL HQ WORKER DETECTION (all true groups pooled) ====")
        print(f"  TP={tp_all}, FP={fp_all}, FN={fn_all}")
        print(f"  Precision={precision_all:.3f}, Recall={recall_all:.3f}, F1={f1_all:.3f}")
        
        
        #########################################
        # 4. Heatmaps of HQ worker identification
        #########################################
        # Build matrices: rows = true groups, cols = workers
        #  - true:  1 if worker is true HQ for that group
        #  - pred:  1 if worker is predicted HQ for that true group (via MV mapping above)
        
        true_hq_matrix = worker_label.T      # shape (n_groups, n_worker)
        
        pred_hq_matrix = np.zeros((n_task_groups, n_worker))
        for h in range(n_task_groups):
            for w in pred_hq_by_true_group[h]:
                pred_hq_matrix[h, w] = 1.0
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(true_hq_matrix, cmap="viridis")
        plt.title("True HQ workers\n(rows=true groups, cols=workers)")
        plt.xlabel("Worker")
        plt.ylabel("True group")
        
        plt.subplot(1, 2, 2)
        sns.heatmap(pred_hq_matrix, cmap="viridis")
        plt.title("Predicted HQ workers\n(rows=true groups, cols=workers)")
        plt.xlabel("Worker")
        plt.ylabel("True group")
        
        plt.tight_layout()
        plt.show()
    
    return task_accuracy, task_label_pred, hq_workers_pred