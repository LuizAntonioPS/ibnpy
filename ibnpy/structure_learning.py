# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:46:33 2021

@author: LuizPereira
"""

# %% Structure Learning
def fit(
        df,
        estimator='hc',
        step_size='100',
        start_dag=None,
        data_order='random',
        max_indegree=None,
        black_list=None,
        white_list=None,
        bw_list_method=None,
        tabu_length=100,
        epsilon=1e-4,
        max_iter=1e6,
        root_node=None):
    """Online structure learning fit model.
    
    Description
    -----------
    
    Parameters
    ----------
    
     Returns
    -------
    
    Examples
    --------
    
    """