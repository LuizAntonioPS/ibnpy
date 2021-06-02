# -*- coding: utf-8 -*-
"""
Created on Thu May 6 20:36:32 2021

@author: LuizPereira

@source: https://github.com/pgmpy/pgmpy/blob/bda1004f3065bf5e25247e5c6be457dce0b1b2b1/pgmpy/estimators/TreeSearch.py
"""

import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from pgmpy.base import UndirectedGraph
from pgmpy.estimators import StructureEstimator
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import (
    mutual_info_score
)

class UndirectedTreeSearch(StructureEstimator):
    def __init__(self, data=None, n_jobs=-1, **kwargs):
        """
        Search class for learning a tree-shaped undirected skeleton. The algorithm
        supported is Chow-Liu[1]. It constructs the maximum-weight spanning tree with
        mutual information score as edge weights.
        Parameters
        ----------
        data: pandas.DataFrame object
            dataframe object where each column represents one variable.
        n_jobs: int (default: -1)
            Number of jobs to run in parallel. `-1` means use all processors.
        References
        ----------
        [1] Chow, C. K.; Liu, C.N. (1968), "Approximating discrete probability
            distributions with dependence trees", IEEE Transactions on Information
            Theory, IT-14 (3): 462–467
        """
        self.data = data
        self.n_jobs = n_jobs

        super(UndirectedTreeSearch, self).__init__(data, **kwargs)

    def estimate(
        self,
        estimator_type="chow-liu",
        edge_weights_fn="mutual_info",
        show_progress=True,
    ):
        """
        Estimate the tree-shaped undirected skeleton that fits best to the given
        data set without parametrization.
        Parameters
        ----------
        estimator_type: str (chow-liu)
            The algorithm to use for estimating the DAG.
        edge_weights_fn: str or function (default: mutual info)
            Method to use for computing edge weights.
        show_progress: boolean
            If True, shows a progress bar for the running algorithm.
        Returns
        -------
        model: `pgmpy.base.UndirectedGraph` instance
            The estimated tree-shaped undirected skeleton.
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import networkx as nx
        >>> from pgmpy.estimators import UndirectedTreeSearch
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> est = UndirectedTreeSearch(values)
        >>> model = est.estimate(estimator_type='chow-liu')
        >>> nx.draw_circular(model, with_labels=True, arrowsize=20, arrowstyle='fancy',
        ...                  alpha=0.3)
        """
        # Step 1. Argument checks
        if estimator_type not in {"chow-liu"}:
            raise ValueError("Invalid estimator_type. Expected only chow-liu.")

        return UndirectedTreeSearch.skeleton_chow_liu(
            self.data, edge_weights_fn, self.n_jobs, show_progress
        )
    
    def skeleton_chow_liu(
        data, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True
    ):
        """
        Chow-Liu algorithm for estimating a tree-shaped undirected skeleton from
        given data.
        Parameters
        ----------
        data: pandas.DataFrame object
            dataframe object where each column represents one variable.
        edge_weights_fn: str or function (default: mutual_info)
            Method to use for computing edge weights. Options are:
                1. 'mutual_info': Mutual Information Score.
                2. 'adjusted_mutual_info': Adjusted Mutual Information Score.
                3. 'normalized_mutual_info': Normalized Mutual Information Score.
        n_jobs: int (default: -1)
            Number of jobs to run in parallel. `-1` means use all processors.
        show_progress: boolean
            If True, shows a progress bar for the running algorithm.
        Returns
        -------
        model: `pgmpy.base.UndirectedGraph` instance
            The estimated tree-shaped undirected skeleton.
            
        """
        # Step 0: Check for edge weight computation method
        if edge_weights_fn == "mutual_info":
            edge_weights_fn = mutual_info_score
        else:
            raise ValueError("edge_weights_fn should either be 'mutual_info'")

        # Step 1: Compute edge weights for a fully connected graph.
        n_vars = len(data.columns)
        if show_progress:
            pbar = tqdm(
                combinations(data.columns, 2), total=(n_vars * (n_vars - 1) / 2)
            )
            pbar.set_description("Building tree")
        else:
            pbar = combinations(data.columns, 2)

        vals = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(edge_weights_fn)(data.loc[:, u], data.loc[:, v]) for u, v in pbar
        )
        weights = np.zeros((n_vars, n_vars))
        weights[np.triu_indices(n_vars, k=1)] = vals
            
        weights = np.zeros((n_vars, n_vars))
        weights[np.triu_indices(n_vars, k=1)] = vals

        # Step 2: Compute the maximum spanning tree using the weights.
        T = nx.maximum_spanning_tree(
            nx.from_pandas_adjacency(
                pd.DataFrame(weights, index=data.columns, columns=data.columns),
                create_using=nx.Graph,
            )
        )
        return UndirectedGraph(T)