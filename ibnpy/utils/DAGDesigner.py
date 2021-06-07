# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:12:48 2021

@author: LuizPereira
"""
import numpy as np
import pandas as pd
import pygraphviz
import tempfile
import matplotlib.pyplot as plt
import matplotlib.image as matimg

def find_edges(df):
    """Finds the edges in the square adjacency matrix, using
    vectorized operations. Returns a list of pairs of tuples
    that represent the edges."""

    values = df.values  # Adjacency matrix of True's and False's
    n_rows, n_columns = values.shape
    indices = np.arange(n_rows*n_columns)
    values = values.flatten()
    aux_indices = indices[values == True]  # True means that the edge exists
    
    # Create two arrays `rows` and `columns` such that for an edge i,
    # (rows[i], columns[i]) is its coordinate in the df
    rows = aux_indices / n_columns
    rows = rows.astype(int)
    columns = aux_indices % n_columns
    
    # Convert the coordinates to actual names
    row_names = df.index[rows]
    column_names = df.columns[columns]
    return zip(row_names, column_names)    # Possible that itertools.izip is faster

def plot(DAG):
    df = _dag2adjmat(model=DAG)
    
    G = pygraphviz.AGraph(directed=True)

    for node in df.index.tolist():
        G.add_node(node, color='blue')

    for parent, child in find_edges(df):
        G.add_edge(parent, child)
    
    with tempfile.NamedTemporaryFile() as tf:
        G.draw(tf, format='jpeg', prog='dot')
        img = matimg.imread(tf)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
    
def _dag2adjmat(model, verbose=3):
    adjmat = None
    if hasattr(model, 'nodes') and hasattr(model, 'edges'):
        adjmat = pd.DataFrame(data=False, index=model.nodes(), columns=model.nodes()).astype('bool')
        # Fill adjmat with edges
        edges = model.edges()
        # Run over the edges
        for edge in edges:
            adjmat.loc[edge[0], edge[1]]=True
        adjmat.index.name='source'
        adjmat.columns.name='target'
    else:
        if verbose>=1: print('[ibnpy] >Could not convert to adjmat because nodes and/or edges were missing.')
    return(adjmat)