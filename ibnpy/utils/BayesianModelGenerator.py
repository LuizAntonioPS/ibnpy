# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:43:47 2021

@author: LuizPereira
"""
from pgmpy.base import DAG
import gzip
from pkg_resources import resource_filename
import random

def empty_dag(df):
    dag = DAG()
    dag.add_nodes_from(df.columns)
    return dag
    
def fully_connected_dag(df):
    dag = empty_dag(df)
    nodes = list(dag.nodes())
    for u in nodes:
        for i in range(nodes.index(u)):
            dag.add_edge(u, nodes[i])
    return dag

def standard_dag(df):
    try:
        model = df.name
        dag = load_model(model)
        return dag
    except:
        raise ValueError("'DataFrame' object has no a correct attribute 'name'")
        
#def naive_dag(df):
        
def random_standard_dag(df):
    dag = standard_dag(df)
    return _remove_random_edges(dag)
        
def random_dag(df):
    dag = fully_connected_dag(df)
    return _remove_random_edges(dag)

def _remove_random_edges(dag):
    edges = dag.edges()
    n_edges = random.randint(1, len(edges)-2)
    out_edges = random.sample(edges, n_edges)
    dag.remove_edges_from(out_edges)
    return dag

def load_model(model_name):
    """
    Fetches the specified model from bnlearn repository and returns a
    pgmpy.model instance.
    Parameter
    ---------
    model: str
        Any model from bnlearn repository (http://www.bnlearn.com/bnrepository).
        Discrete Bayesian Network Options:
            Small Networks:
                1. asia
                2. cancer
                3. earthquake
                4. sachs
                5. survey
            Medium Networks:
                1. alarm
                2. barley
                3. child
                4. insurance
                5. mildew
                6. water
            Large Networks:
                1. hailfinder
                2. hepar2
                3. win95pts
            Very Large Networks:
                1. andes
                2. diabetes
                3. link
                4. munin1
                5. munin2
                6. munin3
                7. munin4
                8. pathfinder
                9. pigs
                10. munin
        Gaussian Bayesian Network Options:
                1. ecoli70
                2. magic-niab
                3. magic-irri
                4. arth150
        Conditional Linear Gaussian Bayesian Network Options:
                1. sangiovese
                2. mehra
    Example
    -------
    >>> from pgmpy.data import get_example_model
    >>> model = get_example_model(model='asia')
    >>> model
    Returns
    -------
    pgmpy.models instance: An instance of one of the model classes in pgmpy.models
                           depending on the type of dataset.
    """
    from pgmpy.readwrite import BIFReader
    from pgmpy.models import BayesianModel
    from pgmpy.factors.discrete import TabularCPD
    
    filenames = {
        "asia": "utils/example_models/asia.bif.gz",
        "cancer": "utils/example_models/cancer.bif.gz",
        "earthquake": "utils/example_models/earthquake.bif.gz",
        "sachs": "utils/example_models/sachs.bif.gz",
        "survey": "utils/example_models/survey.bif.gz",
        "alarm": "utils/example_models/alarm.bif.gz",
        "barley": "utils/example_models/barley.bif.gz",
        "child": "utils/example_models/child.bif.gz",
        "insurance": "utils/example_models/insurance.bif.gz",
        "mildew": "utils/example_models/mildew.bif.gz",
        "water": "utils/example_models/water.bif.gz",
        "hailfinder": "utils/example_models/hailfinder.bif.gz",
        "hepar2": "utils/example_models/hepar2.bif.gz",
        "win95pts": "utils/example_models/win95pts.bif.gz",
        "andes": "utils/example_models/andes.bif.gz",
        "diabetes": "utils/example_models/diabetes.bif.gz",
        "link": "utils/example_models/link.bif.gz",
        "munin1": "utils/example_models/munin1.bif.gz",
        "munin2": "utils/example_models/munin2.bif.gz",
        "munin3": "utils/example_models/munin3.bif.gz",
        "munin4": "utils/example_models/munin4.bif.gz",
        "pathfinder": "utils/example_models/pathfinder.bif.gz",
        "pigs": "utils/example_models/pigs.bif.gz",
        "munin": "utils/example_models/munin.bif.gz",
        "ecoli70": "",
        "magic-niab": "",
        "magic-irri": "",
        "arth150": "",
        "sangiovese": "",
        "mehra": "",
    }
    
    model_name = model_name.lower()
    if model_name not in filenames.keys():
        raise ValueError("dataset should be one of the options")
    if filenames[model_name] == "":
        raise NotImplementedError("The specified dataset isn't supported")

    path = filenames[model_name]
    with gzip.open(resource_filename("pgmpy", path), "rb") as f:
        content = f.read()

    bifmodel=BIFReader(string=content.decode("utf-8"), n_jobs=1)

    model = BayesianModel(bifmodel.variable_edges)
    model.name = bifmodel.network_name
    model.add_nodes_from(bifmodel.variable_names)

    tabular_cpds = []
    for var in sorted(bifmodel.variable_cpds.keys()):
        values = bifmodel.variable_cpds[var]
        cpd = TabularCPD(var, len(bifmodel.variable_states[var]), values,
                         evidence=bifmodel.variable_parents[var],
                         evidence_card=[len(bifmodel.variable_states[evidence_var])
                                        for evidence_var in bifmodel.variable_parents[var]])
        tabular_cpds.append(cpd)

    model.add_cpds(*tabular_cpds)
    model.name = model_name
    return model

# %% Testing
#import bnlearn

#G=bnlearn.import_DAG('asia', CPD=True)
#current_dag=G['model']
#df = bnlearn.sampling(G, n=10)

#F = standard_dag(df)
#df.name = 'asia'
#F = standard_dag(df)
#F.edges()

#I = random_standard_dag(df)
#I.edges()

#J = random_dag(df)
#J.edges()

#G = empty_dag(df)
#G.edges()

#H = fully_connected_dag(df)
#H.edges()
#bnlearn.plot(H)
