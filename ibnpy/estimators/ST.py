# -*- coding: utf-8 -*-
"""
Created on Thu May  6 21:45:29 2021

@author: LuizPereira
"""
from itertools import combinations, chain
from warnings import warn

import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.estimators import StructureEstimator, HillClimbSearch
from ibnpy.estimators import CITests
from ibnpy.estimators.UndirectedTreeSearch import UndirectedTreeSearch

class ST(StructureEstimator):
    def __init__(self,
                 significance=0.99,
                 max_cond_vars=5,
                 max_indegree=None,
                 epsilon=1e-4,
                 max_iter=1e6,
                 **kwargs):
        """
        Class for 
        
        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            
        References
        ----------
        [1] 
        """
        self.significance = significance
        self.max_cond_vars = max_cond_vars
        self.max_indegree = max_indegree
        self.epsilon = epsilon
        self.max_iter = max_iter
        super(ST, self).__init__(data=None, **kwargs)

    def estimate(
        self,
        new_data=None,
        current_dag=None,
        scoring_method=None,
        n_jobs=-1,
        verbose=3
    ):
        old_data = self.data
        self.data = pd.concat([old_data, new_data])
        if old_data is None:
            old_data = new_data
               
        nodes = self.data.columns
        ci_test = CITests.info_chi
        
        # Step 1: Constraint-based Technique
        
        ## Construct skeleton to Heuristic IND
        skeleton = UndirectedTreeSearch(self.data).estimate(show_progress=False)
        
        ## Selecting candidate parents
        graph = nx.complete_graph(n=nodes, create_using=nx.DiGraph)
        graph_edges = list(graph.edges()).copy()
        for (u, v) in graph_edges:
            ## Simple CI test
            aux = 3
            if not ci_test(
                X=u,
                Y=v,
                Z=[],
                data=new_data,
                significance=self.significance
            ):
                aux = 2
                ## Searching for separating set based on ci test
                if not self._separating_set_ci_test(
                        u=u,
                        v=v,
                        graph=graph,
                        ci_test=ci_test,
                        old_data=old_data
                ):
                    aux = 1
                    ## Validating separation with heuristic IND
                    if not self._heuristicIND(
                            data=self.data,
                            X=u,
                            Y=v,
                            skeleton=skeleton,
                            current_dag=current_dag,
                            significance=self.significance
                    ):
                        ## Kepping v as a possible parent
                        if verbose==5: print('keeping', u, '->', v)
                        continue
            ## Removing v of u parent set
            if verbose==5: print('removing', u, '->', v, 'by', aux)
            graph.remove_edge(u, v)
        
        # Step 2: Hill Climbing Search
        ## Getting the white list from parent sets (adjacency_matrix)
        white_list = list(graph.edges())
        if verbose==4: print('white_list =>', white_list)
        
        ## Peform the search
        est = HillClimbSearch(self.data, scoring_method=scoring_method)
        best_model = est.estimate(
            start_dag=current_dag,
            max_indegree=self.max_indegree,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            white_list=white_list,
            show_progress=False
        )
        
        if verbose==4: print('parcial_edges =>', best_model.edges())
        return best_model
    
    
    def _separating_set_ci_test(self, u, v, graph, ci_test, old_data):
        lim_neighbors = 1
        separating_nodes = ()
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in (u, v)]
        ):
            for nodes_set in chain(
                combinations(set(graph.neighbors(u)) - set([v]), lim_neighbors),
                #combinations(set(graph.neighbors(v)) - set([u]), lim_neighbors),
            ):
                if ci_test(
                    X=u,
                    Y=v,
                    Z=nodes_set,
                    data=old_data,
                    significance=self.significance
                ) and ci_test(
                    X=u,
                    Y=v,
                    Z=nodes_set,
                    data=self.data,
                    significance=self.significance
                ) :
                    separating_nodes = nodes_set
                    break
            
            if not len(separating_nodes) == 0:
                break
            
            if lim_neighbors >= self.max_cond_vars:
                warn("Reached maximum number of allowed conditional variables. Exiting")
                break
            
            lim_neighbors += 1
        
        #if not len(separating_set) == 0:
            #print('==========> SEPARATING SET', separating_set)
            
        return not len(separating_nodes) == 0
             
    
    def _heuristicIND(self, data, X, Y, skeleton, current_dag, significance=0.9):
        """
        

        Parameters
        ----------
        data: pandas.DataFrame object
            dataframe object where each column represents one variable.
        X: int, string, hashable object
            A variable name contained in the data set
        Y: int, string, hashable object
            A variable name contained in the data set, different from X
        skeleton: `pgmpy.base.UndirectedGraph` instance.
            A tree-shaped undirected skeleton.
        current_dag: `pgmpy.base.DAG` instance
            A directed graphical model.
        Returns
        -------
        conditional_separation: bool
            if X and Y can be separated, return True.
            Else, return False.

        """
        
        n_d_X, n_d_Y = self._neighbors_on_paths(current_dag, X, Y)
        n_s_X, n_s_Y = self._neighbors_on_paths(skeleton, X, Y)
        c_X = np.concatenate([n_d_X, n_s_X[~np.isin(n_s_X,n_d_X)]])
        c_Y = np.concatenate([n_d_Y, n_s_Y[~np.isin(n_s_Y,n_d_Y)]])
        conditional_separation = False
        for c in (c_X, c_Y):
            conditional_separation = CITests.info_chi(
                X=X,
                Y=Y,
                Z=c.tolist(),
                data=data,
                significance=significance
            )
            if (conditional_separation): break
            while(len(c) > 1):
                s = float("inf")
                chis = np.empty(len(c))
                for i in range(len(c)):
                    new_c = np.delete(c, i)
                    chis[i], _, _ = CITests.info_chi(
                        X=X,
                        Y=Y,
                        Z=new_c.tolist(),
                        boolean=False,
                        data=data,
                        significance=0.9
                    )
                m = np.argmin(chis)
                if (chis[m] <= 0.0):
                    return True
                elif (chis[m] > s):
                    break
                else:
                    s = chis[m];
                    c = np.delete(c, m)
        
        return conditional_separation
    
    
    def _neighbors_on_paths(self, model, X, Y):
        """
        
        Parameters
        ----------
        model : NetworkX graph
            Source model of path(s).
        X : int, string, hashable object
            Source node. Initial node for path(s)
        Y : int, string, hashable object
            Target node(s). Single node or iterable of nodes at which to end
            path.

        Returns
        -------
        n_X: list
            the neighbors of X that are on paths connected X and Y in model
        n_Y: list
            the neighbors of Y that are on paths connected X and Y in model

        """
        n_X = []
        n_Y = []
        for path in nx.all_simple_paths(model, source=X, target=Y):
            if (len(path) > 2):
                n_X.append(path[1])
                n_Y.append(path[-2])
        return np.array(n_X), np.array(n_Y)
        

# %% Testing

#import bnlearn

#G=bnlearn.import_DAG('alarm', CPD=True)
#current_dag=G['model']
#data = bnlearn.sampling(G, n=10000)
#skeleton = ST(data=data).estimate()

#X='VENTALV'
#Y='CATECHOL'