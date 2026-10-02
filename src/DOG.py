import numpy as np

# ----------------------------------------
# Read annotation data
# ----------------------------------------
def get_DOG():
    raw_rating = []
    
    with open("dataset/dogs-1.merged.label.tsv", "r") as f:
        for line in f:
            item, worker, label = line.split()
            raw_rating.append((item, worker, int(label)))
    
    # ----------------------------------------
    # Read ground truth
    # ----------------------------------------
    truth_dict = {}
    
    with open("dataset/dogs-1.merged.truth.tsv", "r") as f:
        for line in f:
            item, truth = line.split()
            truth_dict[item] = int(truth)
    
    # ----------------------------------------
    # Remap item and worker IDs to 0,1,2,...
    # ----------------------------------------
    item_ids = sorted(set(x[0] for x in raw_rating))
    worker_ids = sorted(set(x[1] for x in raw_rating))
    
    item_map = {item: i for i, item in enumerate(item_ids)}
    worker_map = {worker: j for j, worker in enumerate(worker_ids)}
    
    # ----------------------------------------
    # rating = [task, worker, observed label]
    # ----------------------------------------
    rating = np.array([
        [item_map[item], worker_map[worker], label]
        for item, worker, label in raw_rating
    ], dtype=int)
    
    # ----------------------------------------
    # y_true[i] = true label for task i
    # ----------------------------------------
    y_true = np.array([
        truth_dict[item]
        for item in item_ids
    ], dtype=int)
    
    n_task = len(item_ids)
    n_worker = len(worker_ids)
    
    R_obs = np.full((n_task, n_worker), np.nan)
    
    R_obs[
        rating[:, 0],
        rating[:, 1]
    ] = rating[:, 2]
    
    return rating, y_true, R_obs, n_task, n_worker, np.unique(rating[:, 2])

