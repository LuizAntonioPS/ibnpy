# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:46:33 2021

@author: LuizPereira
"""
import pandas as pd
import numpy as np
from ibnpy.utils.InstanceUtils import batch_generator
from ibnpy.utils import BayesianModelGenerator
from ibnpy.helpers import structural_diff
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.base import DAG
import time

def fit(df,
        estimator='ihcs',
        step_length=100,
        start_dag=None,
        data_order=None,
        max_indegree=None,
        significance=0.99,
        nrss=1e6,
        epsilon=1e-4,
        scoretype='bic',
        ci_test='chi_square',
        max_iter=1e6,
        max_cond_vars=1e6,
        distance_measure='euclidean',
        fixed_edges=set(),
        tabu_length=100,
        black_list=None,
        white_list=None,
        data_seed=None,
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
    data_seed: int, (default : None)
        Semente para que o gerador de batch gere sempre os mesmos dados durante todo o processo incremental. Caso none, nenhuma semente é atribuída e os dados alimentados mudam a cada rodada de aprendizagem
    verbose : int, (default : 3)
        Print progress to screen.
        0: NONE, 1: ERROR,  2: WARNING, 3: INFO (default), 4: DEBUG, 5: TRACE
    
     Returns
    -------
    
    Examples
    --------
    
    """
    
    # Step 1.2: Check the start_dag
    if not isinstance(start_dag, DAG):
        if start_dag == 'empty' or start_dag is None:
            start_dag = BayesianModelGenerator.empty_dag(df)
        elif start_dag == 'fully_connected':
            start_dag = BayesianModelGenerator.fully_connected_dag(df)
        elif start_dag == 'random':
            start_dag = BayesianModelGenerator.random_dag(df)
        elif start_dag == 'standard':
            start_dag = BayesianModelGenerator.standard_dag(df)
        elif start_dag == 'random_standard':
            start_dag = BayesianModelGenerator.random_standard_dag(df)
        else:
            raise ValueError(
                "'start_dag' should have a valid value."
            )
    else:
        if not set(start_dag.nodes()) == set(
            df.columns
        ):
            raise ValueError(
                "'start_dag' should be a DAG with the same variables as the data set, or 'None'."
            )
    
    config = {}
    config['estimator'] = estimator.lower()
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
    config['distance_measure'] = distance_measure
    config['fixed_edges'] = fixed_edges
    config['tabu_length'] = tabu_length
    config['black_list'] = black_list
    config['white_list'] = white_list
    config['data_seed'] = data_seed
           
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
                    max_iter=config['max_iter'],
                    fixed_edges=config['fixed_edges'],
                    tabu_length=config['tabu_length'],
                    black_list=config['black_list'],
                    white_list=config['white_list'])
    
    current_dag = config['start_dag']
       
    # Generating batchs
    generator = batch_generator(data=df,
                                step_length=config['step_length'],
                                data_order=config['data_order'],
                                data_seed=config['data_seed'],
                                distance_measure=config['distance_measure'])
    training_batchs = next(generator)
    batch_size = training_batchs.shape[0]
    
    history = {}
    history['F1'] = np.empty(batch_size)
    history['AUC'] = np.empty(batch_size)
    history['BSF'] = np.empty(batch_size)
    history['TIME'] = np.empty(batch_size)
    
    tt = 0
    true_dag = BayesianModelGenerator.standard_dag(df)
    # Incremental process
    for i in range(batch_size):
              
        t = time.process_time()
        
        batch = pd.DataFrame(training_batchs[i, :, :], columns=df.columns, dtype=int)
        
        # Constructing the score method
        scoring_method = _SetScoringType(batch, config['scoretype'])
        
        # Learning the dag
        ## Using Batch Estimator
        if config['estimator']=='hcs':
            batch = pd.DataFrame(np.concatenate(training_batchs[0:i+1, :, :]), columns=df.columns, dtype=int)
            est = _hcs(data=batch,
                       scoring_method=scoring_method)
            current_dag = est.estimate(max_indegree=config['max_indegree'],
                                       epsilon=config['epsilon'],
                                       max_iter=config['max_iter'],
                                       show_progress=False)
        ## Using Online Estimator
        else:
            current_dag = est.estimate(new_data=batch,
                                       current_dag=current_dag,
                                       scoring_method=scoring_method,
                                       verbose=config['verbose'])
        
        elapsed_time = time.process_time()-t
        tt = tt + elapsed_time
        
        _f1, _auc, _bsf = structural_diff.structural_diff_scores(true_dag, current_dag)
        
        # Storage measure history
        history['F1'][i] = _f1
        history['AUC'][i] = _auc
        history['BSF'][i] = _bsf
        history['TIME'][i] = elapsed_time
        
        if verbose==2:
            print('[ibnpy]> Epoch %s/%s [===] %ss' %(i+1, batch_size, elapsed_time))
        elif verbose==3:
            print('[ibnpy]> Epoch %s/%s [===] F1=%s AUC=%s BSF=%s %ss' %(i+1, batch_size, round(_f1, 3), round(_auc,3), round(_bsf, 3), round(elapsed_time, 3)))
           
    if verbose>=2: print('[ibnpy]> Completed Learning [===] %ss' %(round(tt, 3)))
    
    return current_dag, config, history

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
          max_iter=1e6,
          fixed_edges=set(),
          tabu_length=100,
          black_list=None,
          white_list=None):
    
    from ibnpy.estimators.IncrementalHillClimbSearch import IncrementalHillClimbSearch as IHCS
    
    est = IHCS(nrss=nrss,
               max_indegree=max_indegree,
               epsilon=epsilon,
               max_iter=max_iter,
               fixed_edges=fixed_edges,
               tabu_length=tabu_length,
               black_list=black_list,
               white_list=white_list)
    
    return est

# %% HCS Estimator
def _hcs(data,
         scoring_method):
    
    from pgmpy.estimators import HillClimbSearch as HCS
    
    est = HCS(data=data,
              scoring_method=scoring_method)
    
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
