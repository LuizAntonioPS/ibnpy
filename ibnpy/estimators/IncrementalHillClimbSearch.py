# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:57:18 2021

@author: LuizPereira
"""
from pgmpy.estimators import StructureEstimator

class IncrementalHillClimbSearch(StructureEstimator):
    def __init__(self,
                 nrss=1e6,
                 max_indegree=None,
                 epsilon=1e-4,
                 max_iter=1e6):
        """
        Class for 
        
        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            
        References
        ----------
        [1] 
        """
        self.nrss=nrss
        self.max_indegree=max_indegree
        self.epsilon=epsilon
        self.max_iter=max_iter
        super(IncrementalHillClimbSearch, self).__init__()
        
    def estimate(
        self,
        new_data=None,
        current_dag=None,
        score=None,
        n_jobs=-1,
        verbose=3
    ):
        return True
