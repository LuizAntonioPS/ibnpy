# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:43:42 2021

@author: LuizPereira
"""
# %% structure learning
import bnlearn
G=bnlearn.import_DAG('asia', CPD=True)
df = bnlearn.sampling(G, n=1000)

# %%
from ibnpy import incremental_structure_learning

A = incremental_structure_learning.fit(df, estimator='st')

A.edges()

# %% qualquer coisa
import numpy as np
import pandas as pd
from ibnpy.estimators.ST import ST
import networkx as nx
import bnlearn

values = bnlearn.sampling(bnlearn.import_DAG('alarm', CPD=True), n=1000)
est = ST()
S = est.estimate(old_data=values)
len(S)

# %% Testing ST methods
import numpy as np
import pandas as pd
from ibnpy.estimators.ST import ST
import networkx as nx
import bnlearn

values = bnlearn.sampling(bnlearn.import_DAG('alarm', CPD=True), n=10000)
est = ST(data=values)
S = est.estimate()
G=bnlearn.import_DAG('alarm', CPD=True)
ST.heuristicIND(data=None, X='VENTALV', Y='CATECHOL', skeleton=S, current_dag=G['model'])
for path in ST.heuristicIND(data=None, X='VENTALV', Y='CATECHOL', skeleton=S, current_dag=G['model']):
    print(path)



print(len(G))
print(G.edges())
print(G.has_edge('INSUFFANESTH', 'DISCONNECT'))
nx.draw_circular(G, with_labels=True, arrowsize=20, arrowstyle='fancy', alpha=0.3)


from ibnpy.estimators import CITests
critical_value = CITests.chi_square_critical_value(0.99,100)
print(critical_value)



# %% 
#from utils import InstanceSorter
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20,15]  # Set default figure size
import bnlearn
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import entropy

alarm_df = bnlearn.sampling(bnlearn.import_DAG('alarm', CPD=True), n=100000)
andes_df = bnlearn.sampling(bnlearn.import_DAG('andes', CPD=True), n=100000)
asia_df = bnlearn.sampling(bnlearn.import_DAG('asia', CPD=True), n=100000)
sachs_df = bnlearn.sampling(bnlearn.import_DAG('sachs', CPD=True), n=100000)
water_df = bnlearn.sampling(bnlearn.import_DAG('water', CPD=True), n=100000)

alarm_df.info()
alarm_df.describe()
alarm_df['INTUBATION'].value_counts()
andes_df['SNode_155'].value_counts(normalize=True)
asia_df['dysp'].value_counts(normalize=True)
sachs_df['Raf'].value_counts(normalize=True)
water_df['CNON_12_45'].value_counts(normalize=True)

alarm_df[alarm_df['INTUBATION'] == 2].mean()
alarm_df.apply(np.max)
pd.crosstab(alarm_df['HR'], alarm_df['HRBP'], normalize=True)
sns.countplot(x='CATECHOL', hue='ERRCAUTER', data=alarm_df);

corr = alarm_df.corr()
sns.heatmap(corr, annot = True)
print(corr.min())

asia_df.hist(bins=50)
plt.show()
#i_s = InstanceSorter(df)
#sorted_df = i_s.sort('VENTTUBE')


test_asia_df = bnlearn.sampling(bnlearn.import_DAG('asia', CPD=True), n=10)
#score_test_asia_df = test_asia_df.assign(Score=lambda d: d.sum(1))

d = {0 : 0.0000001, 1 : 1, 2 : 2}
kl_asia_df = asia_df.copy()
kl_asia_df['smoke'] = kl_asia_df['smoke'].map(d)
n = 100
split_kl_asia_df = kl_asia_df[:n]
entropy(kl_asia_df.sample(n=n)['smoke'], qk=split_kl_asia_df['smoke'])
