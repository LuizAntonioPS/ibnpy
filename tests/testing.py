from ibnpy.utils import BayesianModelGenerator, InstanceUtils
from ibnpy import incremental_structure_learning
from ibnpy.utils import DAGDesigner
import matplotlib.pyplot as plt
import pandas as pd

# model = BayesianModelGenerator.load_model('survey')
# df = InstanceUtils.sampling(model, 10000)
dataset = 'asia'
df = pd.read_csv('D:/Documentos/Ferramentas/Spyder/ibnpy/tests/datasets/' + dataset + '.csv')
df.name=dataset

data_seed = 5

# %%
step_length = 1000
data_order = None #'ascending'
start_dag = 'empty' #'fully_connected'
max_indegree = 2 #None

# %% Online Learning - Online Estimator - IHCS
max_nrss = len(df.columns)
#max_nrss = max_nrss*(max_nrss-1)
max_nrss = 2

A_IHCS, _, history_IHCS = incremental_structure_learning.fit(df,
                                                             estimator='ihcs',
                                                             nrss=max_nrss,
                                                             max_indegree=max_indegree,
                                                             step_length=step_length,
                                                             start_dag=start_dag,
                                                             data_order=data_order,
                                                             data_seed=data_seed,
                                                             verbose=3)
DAGDesigner.plot(A_IHCS)

# %% Online Learning - Online Estimator
significance = 0.99

A_ST, _, history_ST = incremental_structure_learning.fit(df,
                                                         estimator='st',
                                                         significance=significance,
                                                         max_indegree=max_indegree,
                                                         step_length=step_length,
                                                         start_dag=start_dag,
                                                         data_order=data_order,
                                                         data_seed=data_seed,
                                                         verbose=3)
DAGDesigner.plot(A_ST)

# %% Online Learning - Batch Estimator
A_HCS, _, history_HCS = incremental_structure_learning.fit(df,
                                                           estimator='hcs',
                                                           step_length=step_length,
                                                           start_dag=start_dag,
                                                           data_order=data_order,
                                                           max_indegree=max_indegree,
                                                           data_seed=data_seed,
                                                           verbose=3)
DAGDesigner.plot(A_HCS)

# %% Batch Learning - Batch Estimator
# from pgmpy.estimators import HillClimbSearch
# from pgmpy.estimators import BicScore
# from ibnpy.helpers import structural_diff
# est = HillClimbSearch(df, scoring_method=BicScore(df))
# B_HCS = est.estimate()
# true_dag = BayesianModelGenerator.standard_dag(df)
# _f1, _auc, _bsf = structural_diff.structural_diff_scores(true_dag, B_HCS)
# print('[ibnpy]> Completed Learning [===] F1=%s AUC=%s BSF=%s' %(round(_f1, 3), round(_auc,3), round(_bsf, 3)))
# DAGDesigner.plot(B_HCS)


# %% Plot metric inter algorithm

def plot(metric):
    fig, ax = plt.subplots()
    if metric == 'F1' or metric == 'AUC':
        ax.set(ylim=(0, 1))
    elif metric == 'BSF':
        ax.set(ylim=(-1, 1))
        
    try:
        ax.plot(history_IHCS[metric], label='IHCS')
    except Exception:
        print('IHCS history is not defined')
    
    try:
        ax.plot(history_ST[metric], label='ST')
    except Exception:
        print('ST history is not defined')
    
    try:
        ax.plot(history_HCS[metric], label='HCS')
    except Exception:
        print('HCS history is not defined')
    
    ax.set_title(metric)
    ax.legend()
    plt.show()

plot('F1')
plot('AUC')
plot('BSF')
plot('TIME')

# %% Plot metric inter algorithm

def plot(algorithm):
    
    try:
        if algorithm == 'IHCS':
            history = history_IHCS
        elif algorithm == 'ST':
            history = history_ST
    except Exception as e:
        print(str(e))
        return
    
    fig, ax = plt.subplots()
    ax.set(ylim=(-1, 1))
        
    try:
        ax.plot(history['F1'], label='F1')
        ax.plot(history['AUC'], label='AUC')
        ax.plot(history['BSF'], label='BSF')
    except Exception:
        print(algorithm, 'history is not defined')
    
    ax.set_title(algorithm)
    ax.legend()
    plt.show()

plot('ST')

# %% EXTRA RUN FOR EACH ALGORITHM %% #

# %% Online Learning - Online Estimator - IHCS
A_IHCS_2, _, history_IHCS_2 = incremental_structure_learning.fit(df,
                                                             estimator='ihcs',
                                                             nrss=2,
                                                             #nrss=max_nrss,
                                                             max_indegree=max_indegree,
                                                             step_length=step_length,
                                                             start_dag=start_dag,
                                                             data_order=data_order,
                                                             data_seed=data_seed,
                                                             verbose=3)
DAGDesigner.plot(A_IHCS_2)

# %% Online Learning - Online Estimator
A_ST_2, _, history_ST_2 = incremental_structure_learning.fit(df,
                                                         estimator='st',
                                                         #significance=0.9,
                                                         significance=0.99,
                                                         max_indegree=max_indegree,
                                                         step_length=step_length,
                                                         start_dag=start_dag,
                                                         data_order=data_order,
                                                         data_seed=data_seed,
                                                         verbose=3)
DAGDesigner.plot(A_ST_2)


# %% Plot metric intra algorithm

def plot_2(algorithm, metric):
    
    try:
        if algorithm == 'IHCS':
            h_1 = history_IHCS
            h_2 = history_IHCS_2
        elif algorithm == 'ST':
            h_1 = history_ST
            h_2 = history_ST_2
    except Exception as e:
        print(str(e))
        return
    
    fig, ax = plt.subplots()
    if metric == 'F1' or metric == 'AUC':
        ax.set(ylim=(0, 1))
    elif metric == 'BSF': 
        ax.set(ylim=(-1, 1))
    ax.plot(h_1[metric], label=algorithm)
    ax.plot(h_2[metric], label=algorithm+'_2')
    ax.set_title(metric)
    ax.legend()
    plt.show()
    
plot_2('IHCS', 'F1')
plot_2('IHCS', 'AUC')
plot_2('IHCS', 'BSF')
plot_2('IHCS', 'TIME')
plot_2('ST', 'F1')
plot_2('ST', 'AUC')
plot_2('ST', 'BSF')
plot_2('ST', 'TIME')