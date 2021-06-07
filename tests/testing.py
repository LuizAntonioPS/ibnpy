from ibnpy.utils import BayesianModelGenerator, InstanceUtils
from ibnpy import incremental_structure_learning
from ibnpy.utils import DAGDesigner

model = BayesianModelGenerator.load_model('cancer')
df = InstanceUtils.sampling(model, 100000)

A = incremental_structure_learning.fit(df, estimator='st', start_dag='empty', verbose=3, step_length=1000)
#print(A.edges())
DAGDesigner.plot(A)

#A = incremental_structure_learning.fit(df, estimator='st', start_dag='empty', verbose=3, step_length=100)
#DAGDesigner.plot(A)
#B = incremental_structure_learning.fit(df, estimator='st', start_dag='empty', verbose=3, step_length=1000)
#DAGDesigner.plot(B)
#C = incremental_structure_learning.fit(df, estimator='st', start_dag='fully_connected', verbose=3, step_length=100)
#DAGDesigner.plot(C)
#D = incremental_structure_learning.fit(df, estimator='st', start_dag='fully_connected', verbose=3, step_length=1000)
#DAGDesigner.plot(D)