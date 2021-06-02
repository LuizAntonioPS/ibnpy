# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:46:33 2021

@author: LuizPereira
"""
import pandas as pd
from ibnpy.utils.InstanceUtils import batch_generator
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.base import DAG
import time

def fit(df,
        estimator='ihcs',
        step_length=100,
        start_dag=None,
        data_order='random',
        max_indegree=None,
        epsilon=1e-4,
        scoretype='bic',
        ci_test='chi_square',
        significance=0.99,
        nrss=1e6,
        max_iter=1e6,
        max_cond_vars=1e6,
        verbose=3):
    """Online structure learning fit model.
    
    Args:
    
    Parameters #TODO
    ----------
    df : pd.DataFrame()
        Input dataframe.
    estimator : str, (default : 'ihcs')
        String Search strategy for incremental structure_learning.
        'ihcs' or 'incrementalhillclimbsearch' (default)
        'st' or 'shitansearch'
    scoretype : str, (default : 'bic')
        Scoring function for the search spaces.
        'bic', 'k2', 'bdeu'
    max_indegree : int, (default : None)
        If provided and unequal None, the procedure only searches among models where all nodes have at most max_indegree parents. (only in case of methodtype='hc')
    epsilon: float (default: 1e-4)
        Defines the exit condition. If the improvement in score is less than `epsilon`, the learned model is returned. (only in case of methodtype='hc')
    max_iter: int (default: 1e6)
        The maximum number of iterations allowed. Returns the learned model when the number of iterations is greater than `max_iter`. (only in case of methodtype='hc')
    verbose : int, (default : 3)
        Print progress to screen.
        0: NONE, 1: ERROR,  2: WARNING, 3: INFO (default), 4: DEBUG, 5: TRACE
    
     Returns
    -------
    
    Examples
    --------
    
    """
    
    # Step 1.2: Check the start_dag
    if start_dag is None:
        start_dag = DAG()
        start_dag.add_nodes_from(df.columns)
    elif not isinstance(start_dag, DAG) or not set(start_dag.nodes()) == set(
        df.columns
    ):
        raise ValueError(
            "'start_dag' should be a DAG with the same variables as the data set, or 'None'."
        )
    
    config = {}
    config['estimator'] = estimator
    config['step_length'] = step_length
    config['start_dag'] = start_dag
    config['data_order'] = data_order
    config['max_indegree'] = max_indegree
    config['epsilon'] = epsilon
    config['scoretype'] = scoretype
    config['nrss'] = nrss
    config['ci_test'] = ci_test
    config['significance'] = significance
    config['max_iter'] = max_iter
    config['max_cond_vars'] = max_cond_vars
    config['verbose'] = verbose
        
    if config['estimator']=='st':
        est = _st(significance=config['significance'],
                  max_cond_vars=config['max_cond_vars'],
                  max_indegree=config['max_indegree'],
                  epsilon=config['epsilon'],
                  max_iter=config['max_iter'])
        
    if config['estimator']=='ihcs':
        est = _ihcs(nrss=config['nrss'],
                    max_indegree=config['max_indegree'],
                    epsilon=config['epsilon'],
                    max_iter=config['max_iter'])
    
    current_dag = config['start_dag']
    
    # Generating batchs #TODO falta data_order
    generator = batch_generator(data=df,
                                step_length=config['step_length'],
                                data_order=config['data_order'])
    training_batchs = next(generator)
    batch_size = training_batchs.shape[0]
    
    # Incremental process
    for i in range(batch_size):
        
        t = time.process_time()
        
        batch = pd.DataFrame(training_batchs[i, :, :], columns=df.columns, dtype=float)
        
        # Constructing the score method
        scoring_method = _SetScoringType(batch, config['scoretype'])
        
        # Learning the dag
        current_dag = est.estimate(new_data=batch,
                                   current_dag=current_dag,
                                   scoring_method=scoring_method,
                                   verbose=config['verbose'])
        
        elapsed_time = time.process_time()-t
        
        if verbose>=3: print('[ibnpy] >Epoch %s/%s [===] %ss' %(i+1, batch_size, elapsed_time))
    
    return current_dag

# %% ST Estimator
def _st(significance=0.99,
        max_cond_vars=5,
        max_indegree=None,
        epsilon=1e-4,
        max_iter=1e6):
    
    from ibnpy.estimators import ST
    
    est = ST(significance=significance,
             max_cond_vars=max_cond_vars,
             max_indegree=max_indegree,
             epsilon=epsilon,
             max_iter=max_iter)
    
    return est

# %% IHCS Estimator
def _ihcs(nrss=1e6,
          max_indegree=None,
          epsilon=1e-4,
          max_iter=1e6):
    
    from ibnpy.estimators.IncrementalHillClimbSearch import IncrementalHillClimbSearch as IHCS
    
    est = IHCS(nrss=nrss,
               max_indegree=max_indegree,
               epsilon=epsilon,
               max_iter=max_iter)
    
    return est

# %% Set scoring type
def _SetScoringType(df, scoretype):

    if scoretype=='bic':
        scoring_method = BicScore(df)
    elif scoretype=='k2':
        scoring_method = K2Score(df)
    elif scoretype=='bdeu':
        scoring_method = BDeuScore(df, equivalent_sample_size=5)

    return(scoring_method)
