# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 18:02:19 2026

@author: wangl
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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
    biased_workers_pred = [None] * n_task_groups
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
    
       
            
        eigvals, eigvecs = np.linalg.eigh(agreement_g)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # top-2 eigenvectors
        V2 = eigvecs[:, :2]  # shape: (n_worker, 2)
        
        # optional: normalize each eigenvector (usually unnecessary)
        V2 = V2 / (np.linalg.norm(V2, axis=0, keepdims=True) + 1e-12)
        
        scores = np.linalg.norm(V2, axis=1)  # shape: (n_worker,)
        
        scores_reshaped = scores.reshape(-1, 1)  # (n_worker, 1)

        km = KMeans(n_clusters=3, n_init=10)
        labels_np = km.fit_predict(scores_reshaped)
        centers_np = km.cluster_centers_.flatten()  # (3,)

        # Identify which cluster label corresponds to which archetype
        # by sorting cluster centers: smallest=LQ, largest=HQ, middle=biased
        order = np.argsort(centers_np)  # order[0]=LQ, order[1]=biased, order[2]=HQ

        lq_label     = order[0]
        biased_label = order[1]
        hq_label     = order[2]

        hq_workers_g     = np.where(labels_np == hq_label)[0]
        biased_workers_g = np.where(labels_np == biased_label)[0]
        lq_workers_g     = np.where(labels_np == lq_label)[0]

        hq_workers_pred[g]      = hq_workers_g
        biased_workers_pred[g]  = biased_workers_g
            
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
    # 3. Worker identification accuracy
    #########################################
    if verbose:
        archetype_names = ['LQ', 'HQ', 'Biased']

        # Build true V from worker_type (n_worker, n_task_groups, 3)
        # worker_type[j, g, t] = 1 iff worker j is archetype t for group g
        V_true = np.argmax(worker_label, axis=2)  # (n_worker, n_task_groups)

        # Build predicted V from hq_workers_pred and biased_workers_pred
        # 0=LQ (default), 1=HQ, 2=biased
        V_pred = np.zeros((n_worker, n_task_groups), dtype=int)
        for g in range(n_task_groups):
            if hq_workers_pred[g] is not None:
                V_pred[hq_workers_pred[g], g] = 1
            if biased_workers_pred[g] is not None:
                V_pred[biased_workers_pred[g], g] = 2

        overall_true = []
        overall_pred = []

        fig, axes = plt.subplots(1, n_task_groups,
                                 figsize=(4 * n_task_groups, 4))
        if n_task_groups == 1:
            axes = [axes]

        print("\n==== WORKER IDENTIFICATION ACCURACY ====")

        for g in range(n_task_groups):
            y_true = V_true[:, g]
            y_pred = V_pred[:, g]

            overall_true.extend(y_true)
            overall_pred.extend(y_pred)

            # ── Confusion matrix ──
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=archetype_names,
                        yticklabels=archetype_names,
                        ax=axes[g])
            axes[g].set_title(f'Task group {g}')
            axes[g].set_xlabel('Predicted')
            axes[g].set_ylabel('True')

            # ── Per-archetype metrics ──
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1, 2],
                average=None, zero_division=0
            )
            print(f"\nTask group {g}:")
            print(f"  {'Archetype':<10} {'Precision':<12} {'Recall':<10} {'F1':<8}")
            for t, name in enumerate(archetype_names):
                print(f"  {name:<10} {prec[t]:<12.3f} {rec[t]:<10.3f} {f1[t]:<8.3f}")

            # ── Critical error rates ──
            bias_as_hq = cm[2, 1] / (cm[2, :].sum() + 1e-12)
            hq_as_bias = cm[1, 2] / (cm[1, :].sum() + 1e-12)
            print(f"  Biased-as-HQ rate : {bias_as_hq:.3f}")
            print(f"  HQ-as-Biased rate : {hq_as_bias:.3f}")

        plt.suptitle("Worker identification confusion matrices", y=1.02)
        plt.tight_layout()
        plt.show()

        # ── Overall pooled confusion matrix ──
        overall_true = np.array(overall_true)
        overall_pred = np.array(overall_pred)

        cm_overall = confusion_matrix(overall_true, overall_pred, labels=[0, 1, 2])
        prec_all, rec_all, f1_all, _ = precision_recall_fscore_support(
            overall_true, overall_pred, labels=[0, 1, 2],
            average=None, zero_division=0
        )

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues',
                    xticklabels=archetype_names,
                    yticklabels=archetype_names,
                    ax=ax2)
        ax2.set_title('Overall pooled confusion matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        plt.tight_layout()
        plt.show()

        print("\n==== OVERALL POOLED METRICS ====")
        print(f"  {'Archetype':<10} {'Precision':<12} {'Recall':<10} {'F1':<8}")
        for t, name in enumerate(archetype_names):
            print(f"  {name:<10} {prec_all[t]:<12.3f} {rec_all[t]:<10.3f} {f1_all[t]:<8.3f}")
        print(f"\n  Overall membership accuracy: {np.mean(overall_true == overall_pred):.3f}")
        
        #########################################
        # 4. Heatmaps of HQ worker identification
        #########################################
        # Build matrices: rows = true groups, cols = workers
        #  - true:  1 if worker is true HQ for that group
        #  - pred:  1 if worker is predicted HQ for that true group (via MV mapping above)
        
        
        #true_hq_matrix = worker_label.T      # shape (n_groups, n_worker)      
        true_hq_matrix = worker_label[:, :, 0].T   # (n_task_groups, n_worker), HQ only
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(true_hq_matrix, cmap="viridis")
        plt.title("True HQ workers\n(rows=true groups, cols=workers)")
        plt.xlabel("Worker")
        plt.ylabel("True group")
        
        
        #########################################
        # 4. Heatmaps of worker agreement per task group
        #########################################
        fig, axes = plt.subplots(1, n_task_groups, figsize=(5 * n_task_groups, 5))
        if n_task_groups == 1:
            axes = [axes]
        
        for g in range(n_task_groups):
            tasks_g = np.where(label == g)[0]
            if len(tasks_g) == 0:
                continue
        
            R_g = R_obs[tasks_g, :]
            agreement_g = np.zeros((n_worker, n_worker))
            for w1 in range(n_worker):
                for w2 in range(n_worker):
                    valid = ~np.isnan(R_g[:, w1]) & ~np.isnan(R_g[:, w2])
                    if valid.sum() == 0:
                        agreement_g[w1, w2] = 0
                    else:
                        agreement_g[w1, w2] = np.mean(R_g[valid, w1] == R_g[valid, w2])
        
            sns.heatmap(agreement_g, cmap="viridis", vmin=0, vmax=1,
                        ax=axes[g], xticklabels=False, yticklabels=False)
            axes[g].set_title(f"Task group {g}\n({len(tasks_g)} tasks)")
            axes[g].set_xlabel("Worker")
            axes[g].set_ylabel("Worker")
        
        plt.suptitle("Worker agreement matrices by predicted task group", y=1.02)
        plt.tight_layout()
        plt.show()
    
    return task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred