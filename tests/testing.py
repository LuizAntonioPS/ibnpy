import bnlearn
G=bnlearn.import_DAG('alarm', CPD=True)
df = bnlearn.sampling(G, n=1000)

from ibnpy import incremental_structure_learning

A = incremental_structure_learning.fit(df, estimator='st', step_length=100, verbose=4)

print(A.edges())