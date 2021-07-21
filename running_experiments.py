# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:25:01 2021

@author: LuizPereira
"""
from ibnpy import incremental_structure_learning
from ibnpy.utils import DAGDesigner
import matplotlib.pyplot as plt
import pandas as pd

#%% Running IHCS experiment

treatments_IHCS = open('D:/Documentos/Ferramentas/Spyder/ibnpy/experiments/treatments.txt', 'r')
treatments = treatments_IHCS.readlines()
treatments = [i for i in treatments if not i.startswith('#')]
 
count = 1
for treatment in treatments:
    if treatment.startswith('#'): continue
    print('=== RUNNING', count,'/',str(len(treatments)),'===') #TODO
    levels = treatment.split(';')
    algorithm = levels[0]
    dataset = levels[1]
    df = pd.read_csv('D:/Documentos/Ferramentas/Spyder/ibnpy/experiments/datasets/' + dataset + '.csv')
    df.name=dataset
    step_length = int(levels[2])
    data_order = None if levels[3] == 'None' else levels[3]
    start_dag = levels[4]
    max_indegree = None if levels[5] == 'None' else int(levels[5])
    max_nrss = len(df.columns)*(len(df.columns)-1) if levels[6] == 'None' else 2
    significance = 0.0 if levels[6] == 'None' else float(levels[6])
    
      
    model, _, history = incremental_structure_learning.fit(df,
                                                     estimator=algorithm,
                                                     nrss=max_nrss,
                                                     significance=significance,
                                                     max_indegree=max_indegree,
                                                     step_length=step_length,
                                                     start_dag=start_dag,
                                                     data_order=data_order,
                                                     verbose=3)
    
    DAGDesigner.plot(model)
    
    (pd.DataFrame.from_dict(data=history)).to_csv('D:/Documentos/Ferramentas/Spyder/ibnpy/experiments/results/'+str(count)+'.csv', sep=';', index=False)
    count += 1
   




# %% Reading results
import matplotlib.pyplot as plt
df_results = pd.read_csv('D:/Documentos/Ferramentas/Spyder/ibnpy/experiments/results/2.csv', sep=';')
def plot(metric):
    fig, ax = plt.subplots()
    if metric == 'F1' or metric == 'AUC':
        ax.set(ylim=(0, 1))
    elif metric == 'BSF':
        ax.set(ylim=(-1, 1))
        
    try:
        ax.plot(df_results[metric], label='IHCS')
    except Exception:
        print('IHCS history is not defined')
    
    ax.set_title(metric)
    ax.legend()
    plt.show()

plot('F1')