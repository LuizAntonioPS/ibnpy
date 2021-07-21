# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:57:18 2021

@author: LuizPereira
"""
from pgmpy.estimators import StructureEstimator
import networkx as nx
from itertools import permutations
from collections import deque
from tqdm import trange
import pandas as pd
from pgmpy.estimators import BicScore

class IncrementalHillClimbSearch(StructureEstimator):
    def __init__(self,
                 nrss=1e6,
                 max_indegree=None,
                 epsilon=1e-4,
                 max_iter=1e6,
                 fixed_edges=set(),
                 tabu_length=100,
                 black_list=None,
                 white_list=None,
                 **kwargs):
        """ #TODO
        Class for 
        
        Parameters
        ----------
        fixed_edges: iterable
            A list of edges that will always be there in the final learned model.
            The algorithm will add these edges at the start of the algorithm and
            will never change it.
        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.
        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.
        black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None
        white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None
        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.
        max_iter: int (default: 1e6)
            The maximum number of iterations allowed. Returns the learned model when the
            number of iterations is greater than `max_iter`.
            
        References
        ----------
        [1] 
        """
        self.search_path=list()
        self.search_space=list()
        
        
        self.nrss=nrss
        self.m_ini=None
        self.d_t=None
        self._search_space_flag=False #usar isto. Setar para true lá no TOCO e atualizar para false novamente depois da 1 busca no HCS após corrigir M_init
        self.max_indegree=max_indegree
        self.epsilon=epsilon
        self.max_iter=max_iter
        self.fixed_edges=fixed_edges
        self.tabu_length=100
        self.black_list=black_list
        self.white_list=white_list
        super(IncrementalHillClimbSearch, self).__init__(data=None, **kwargs)
        
    def estimate(
        self,
        new_data=None,
        current_dag=None,
        scoring_method=None,
        n_jobs=-1,
        verbose=3
    ):
        self.variables=new_data.columns.values
        if self.m_ini is None:
            self.m_ini = current_dag
        
        #Step x.x: Store data
        old_data = self.d_t
        self.d_t = pd.concat([old_data, new_data], ignore_index=True)
        if old_data is None:
            old_data = new_data
        
        #Step x.x: update score method
        scoring_method = BicScore(self.d_t)
        
        #Step 1.2: Check the start_dag
        if not set(current_dag.nodes()) == set(
            list(new_data.columns.values)
        ):
            raise ValueError(
                "'start_dag' should be a DAG with the same variables as the data set, or 'None'."
            )
        #Using TOCO Heuristic to update the initial model
        current_dag = self._toco(scoring_method=scoring_method, current_dag=current_dag)

        # Step 1.3: Check fixed_edges
        fixed_edges=self.fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            fixed_edges = set(fixed_edges)
            current_dag.add_edges_from(fixed_edges)
            if not nx.is_directed_acyclic_graph(current_dag):
                raise ValueError(
                    "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
                )

        # Step 1.4: Check black list and white list
        # user_white_list = (
        #     set()
        #     if self.white_list is None
        #     else set(self.white_list)
        # )
        # search_space = set(sum(self.search_space, ())) # nRSS tá errado, acho.
        #                                                # a vizinhaça de cada modelo tem que ser restrita ao search_space daquele modelo no passo anterior
        # white_list = (
        #     set([(u, v) for u in self.variables for v in self.variables])
        #     if not user_white_list and not search_space
        #     else user_white_list.union(search_space)
        # )
        black_list = set() if self.black_list is None else set(self.black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if self.white_list is None
            else set(self.white_list)
        )
        
        # Step 1.5: Initialize max_indegree, tabu_list, search_path and progress bar
        max_indegree=self.max_indegree
        tabu_length=self.tabu_length
        max_iter=self.max_iter
        nrss=int(self.nrss)
        if max_indegree is None:
	            max_indegree = float("inf")
        tabu_list = deque(maxlen=tabu_length)
        #search_path = []
        #list_neighBorhood = []
        if verbose >= 4:
            iteration = trange(int(max_iter))
        else:
            iteration = range(int(max_iter))
        	
        # Step 2: For each iteration, find the best scoring operation and	
        #         do that to the current model. If no legal operation is	
        #         possible, sets best_operation=None.
        score_fn = scoring_method.local_score
        for _ in iteration:	
            local_search_space=list()
            for value in self._legal_operations(	
                    current_dag,
                    score_fn,
                    tabu_list,
                    max_indegree,
                    black_list,
                    white_list,
                    fixed_edges,
                  ):  	
                  local_search_space.append(value)
            best_operation = None
            if local_search_space:
                local_search_space.sort(key=lambda t: t[1], reverse=True)
                best_operation = local_search_space[0][0]
                best_score_delta = local_search_space[0][1]
            
            if best_operation is None or best_score_delta < self.epsilon:
                break
            else:
                local_search_space=tuple(x for x, y in local_search_space[0:nrss+1])
                if self._search_space_flag:
                    self.search_space = self.search_space[:-1] + [local_search_space]
                    self._search_space_flag = False
                else:
                    self.search_space.append(local_search_space)
                self.search_path.append(best_operation)
                if best_operation[0] == "+":
                    current_dag.add_edge(*best_operation[1])
                    tabu_list.append(("-", best_operation[1]))
                elif best_operation[0] == "-":
                    current_dag.remove_edge(*best_operation[1])
                    tabu_list.append(("+", best_operation[1]))
                elif best_operation[0] == "flip":
                    X, Y = best_operation[1]
                    current_dag.remove_edge(X, Y)
                    current_dag.add_edge(Y, X)
                    tabu_list.append(best_operation)
        # Step 3: Return if no more improvements or maximum iterations reached.
        return current_dag
    
           
    def _toco(
            self,
            scoring_method=None,
            current_dag=None
    ):
        model = self.m_ini.copy()
        for i, neighborhood_operators in enumerate(self.search_space):
            search_space = tuple(set(neighborhood_operators).intersection(set(self.search_path)))
            local_search_space = list()
            for operation in search_space:
                score = self._operation_score(model, operation, scoring_method)
                local_search_space.append((operation, score))
                
            if not local_search_space:
                self.search_path = self.search_path[:i]
                # self.search_space = self.search_space[:i]
                self.search_space = (
                    self.search_space[:i+1]
                    if i != 0
                    else list()
                )
                self._search_space_flag = len(self.search_space) != 0
                break
                
            local_search_space.sort(key=lambda t: t[1], reverse=True)
            best_local_operation = local_search_space[0][0]
                
            if local_search_space[0][1] > self.epsilon and best_local_operation[0] == self.search_path[i][0] and best_local_operation[1] == self.search_path[i][1]:
                self._do_operation(model, self.search_path[i])
            else:
                self.search_path = self.search_path[:i]
                # self.search_space = self.search_space[:i]
                self.search_space = (
                    self.search_space[:i+1]
                    if i != 0
                    else list()
                )
                self._search_space_flag = len(self.search_space) != 0
                break
        
        return model
            
    
    def _undo_operation_score(self, model, operation, scoring_method):
        if operation[0] == "+":
            operation[0] = "-"
        elif operation[0] == "-":
            operation[0] = "+"
        elif operation[0] == "flip":
            X, Y = operation[1]
            operation[1] = (Y, X)
            
        return self._operation_score(model, operation, scoring_method)
      
    def _operation_score(self, model, operation, scoring_method):
        score = scoring_method.local_score
        X, Y = operation[1]
        old_Y_parents = model.get_parents(Y)     
        new_Y_parents = old_Y_parents[:]
        
        if operation[0] == "flip":
            old_X_parents = model.get_parents(X)
            new_X_parents = old_X_parents[:]
            # Check if flipping creates any cycles
            if nx.has_path(model, X, Y) and not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                new_X_parents = old_X_parents + [Y]
                new_Y_parents.remove(X)
            score_delta = (
                    score(X, new_X_parents)
                    + score(Y, new_Y_parents)
                    - score(X, old_X_parents)
                    - score(Y, old_Y_parents)
                )
        else:
            if operation[0] == "+":
                if not nx.has_path(model, X, Y):
                    new_Y_parents = new_Y_parents + [X]
            elif operation[0] == "-":
                #if nx.has_path(model, X, Y):
                if X in new_Y_parents:
                    new_Y_parents.remove(X)
            score_delta = score(Y, new_Y_parents) - score(Y, old_Y_parents)
            
        return score_delta
    
    def _undo_operations(self, current_model, operations):
        for operation in operations:
            if operation[0] == "+":
                current_model.remove_edge(*operation[1])
            elif operation[0] == "-":
                current_model.add_edge(*operation[1])
            elif operation[0] == "flip":
                X, Y = operation[1]
                current_model.remove_edge(Y, X)
                current_model.add_edge(X, Y)
            
    def _do_operation(self, current_model, operation):
        if operation[0] == "+":
            current_model.add_edge(*operation[1])
        elif operation[0] == "-":
            current_model.remove_edge(*operation[1])
        elif operation[0] == "flip":
            X, Y = operation[1]
            current_model.remove_edge(X, Y)
            current_model.add_edge(Y, X)
            
    def _is_neighborhood(self, operation):
        if not self.d_t is None:
            if self.search_path:
                if self._search_space_flag:
                    local_search_space = self.search_space[-1:]
                    local_search_space = local_search_space[0]
                    t = operation in local_search_space
                    return t
        return True
            
    def _legal_operations(
        self, model, score, tabu_list, max_indegree, black_list, white_list, fixed_edges
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )
        
        for (X, Y) in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                    (operation not in tabu_list)
                    and ((X, Y) not in black_list)
                    and ((X, Y) in white_list)
                    and self._is_neighborhood(operation)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for (X, Y) in model.edges():
            operation = ("-", (X, Y))
            if (operation not in tabu_list) and ((X, Y) not in fixed_edges) and self._is_neighborhood(operation):
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)
                score_delta = score(Y, new_parents) - score(Y, old_parents)
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for (X, Y) in model.edges():
            # Check if flipping creates any cycles
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                    and ((X, Y) not in fixed_edges)
                    and ((Y, X) not in black_list)
                    and ((Y, X) in white_list)
                    and self._is_neighborhood(operation)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                            score(X, new_X_parents)
                            + score(Y, new_Y_parents)
                            - score(X, old_X_parents)
                            - score(Y, old_Y_parents)
                        )
                        yield (operation, score_delta)