# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:29:04 2021

@author: LuizPereira
"""
def _independencies(true_dag, a):
    _n = len(true_dag.nodes())
    maximum_edges = _n*(_n-1)/2
    return maximum_edges-a;

def _specificity(tn, fp):
    return tn/(tn+fp+1e-4)

def _recall(tp, fn):
    return tp/(tp+fn+1e-4)

def _precision(tp, fp):
    return tp/(tp+fp+1e-4)

def _f1(precision, recall):
    return 2*precision*recall/(precision+recall+1e-4)
    
def _auc(recall, specificity):
    return (recall+specificity)/2

def _bsf(tp, fp, tn, fn, a, i):
    return ((tp/a)+(tn/i)-(fp/i)-(fn/a))/2

def structural_diff_scores(true_dag, predict_dag):
    _tp, _tp_c, tp_b, _fp, _tn, _fn, _a, _i = _cm(true_dag, predict_dag)
    _p = _precision(_tp, _fp)
    _r = _recall(_tp, _fn)
    _s = _specificity(_tn, _fp)
    f1 = _f1(_p, _r)
    auc = _auc(_r, _s)
    bsf = _bsf(_tp, _fp, _tn, _fn, _a, _i)
    
    return f1, auc, bsf


def _cm(true_dag, predict_dag):
    tp = 0
    tp_b = 0
    tp_c = 0
    tn = 0
    fp = 0
    fn = 0
    a = 0
    nodes = list(true_dag.nodes())
    for u in nodes:
        for v in nodes:
            if nodes.index(v) > nodes.index(u):
                true_edge_learnt_graph = v in predict_dag.get_parents(u)
                reversed_edge_learnt_graph = u in predict_dag.get_parents(v)
                true_edge_ground_truth_graph = v in true_dag.get_parents(u)
                reversed_edge_ground_truth_graph = u in true_dag.get_parents(v)
                no_edge_learnt_graph = not true_edge_learnt_graph and not reversed_edge_learnt_graph
                no_edge_ground_truth_graph = not true_edge_ground_truth_graph and not reversed_edge_ground_truth_graph
                
                if no_edge_learnt_graph and no_edge_ground_truth_graph: tn=tn+1
                elif no_edge_learnt_graph and not no_edge_ground_truth_graph: fn=fn+1
                elif not no_edge_learnt_graph and no_edge_ground_truth_graph: fp=fp+1
                elif not no_edge_learnt_graph and not no_edge_ground_truth_graph:
                    if true_edge_learnt_graph and reversed_edge_ground_truth_graph: tp_b=tp_b+1
                    if reversed_edge_learnt_graph and true_edge_ground_truth_graph: tp_b=tp_b+1
                    if true_edge_learnt_graph and true_edge_ground_truth_graph: tp_c=tp_c+1
                    if reversed_edge_learnt_graph and reversed_edge_ground_truth_graph: tp_c=tp_c+1
					
                if true_edge_ground_truth_graph or reversed_edge_ground_truth_graph: a = a+1
    
    tp = tp_c+tp_b/2
    fp = fp+(tp_c+tp_b-tp)
    i = _independencies(true_dag, a)      
    return tp, tp_c, tp_b, fp, tn, fn, a, i


# %% Testing
#import bnlearn
#from ibnpy.utils import BayesianModelGenerator

#G=bnlearn.import_DAG('asia', CPD=True)
#true = G['model']
#df = bnlearn.sampling(G, n=2)
#df.name = 'asia'
#predict = BayesianModelGenerator.random_standard_dag(df)
#print(structural_diff_scores(true, predict))
