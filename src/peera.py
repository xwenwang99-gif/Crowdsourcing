# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 13:49:20 2025

@author: wangl
"""

import peerannot.models as pa
import numpy as np
from collections import defaultdict

class peerA():
    def __init__(self, rating, task_group,n_worker):
        self.rating = rating
        self.task_group = task_group
        self.n_worker = n_worker
    
    def _dict(self):
        task_dict = defaultdict(dict)
        
        for task, worker, label in self.rating:
            task = int(float(task))    # handles "0", "0.0", etc.
            worker = int(float(worker))
            label = int(float(label))

            task_dict[f"{task}"][f"{worker}"] = label
            
        task_dict = dict(task_dict)
        
        return task_dict
    
    def _MV(self):
        task_dict = self._dict()
        MV_model = pa.MV(task_dict, self.task_group, sparse=False)
        return MV_model.get_answers()
    
    def _GLAD(self):
        task_dict = self._dict()
        GLAD_model = pa.GLAD(task_dict, n_classes=self.task_group,n_workers=self.n_worker)
        return GLAD_model.get_answers(), GLAD_model.get_probas()
    