# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:17:46 2021

@author: LuizPereira
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, cosine

def _append_score(df, measure='cityblock'):
    measure = _SetDistanceMeasureType(measure)
    base_row = df.iloc[0]
    rows = df.iloc[1:]
    scores = np.zeros(len(df))
    for i in range(len(rows)):
        row = rows.iloc[i]
        scores[i+1] = measure(base_row, row)
    
    df['score'] = pd.Series(scores)
    
    return df

def _sorter(df, data_order=None, distance_measure='cityblock'):
    if not data_order is None:
        df = _append_score(df, measure=distance_measure)
        ascending = data_order == 'ascending'
        df = df.sort_values(by=['score'], ascending=ascending)
        df = df.drop(columns=['score'])
    return df

    
def batch_generator(data,
                    data_order=None,
                    features=None,
                    classes=None,
                    batch_size=None,
                    step_length=100,
                    distance_measure='cityblock'
                    ):
    """
    Generator function for creating batches of training-data.

    """
    
    data = _sorter(df=data, data_order=data_order, distance_measure=distance_measure)
       
    if not features is None and not classes is None:
        x = data[features]
        y = data[classes]
    else:
        x = data
    
    if batch_size is None:
        batch_size = int(len(data)/step_length)
        
    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-data.
        x_shape = (batch_size, step_length, len(x.columns))
        x_batch = np.zeros(shape=x_shape, dtype=np.int16)

        # Allocate a new array for the batch of output-data.
        if not classes is None:
            y_shape = (batch_size, step_length, len(y.columns))
            y_batch = np.zeros(shape=y_shape, dtype=np.int16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = 0
            if len(x) - step_length != 0:
                idx = np.random.randint(len(x) - step_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x[idx:idx+step_length]
            if not classes is None:
                y_batch[i] = y[idx:idx+step_length]
        
        if not classes is None:
            yield (x_batch, y_batch)
        else:
            yield x_batch
            
# %% Set scoring type
def _SetDistanceMeasureType(measure):

    if measure=='euclidean':
        distance_measure = euclidean
    elif measure=='cityblock':
        distance_measure = cityblock
    elif measure=='cosine':
        distance_measure = cosine

    return(distance_measure)

# %% Sampling
from pgmpy.sampling import BayesianModelSampling
def sampling(model, n=1000):
    infer_model = BayesianModelSampling(model)
    df=infer_model.forward_sample(size=n, return_type='dataframe')
    df.name=model.name
    return df
            
# %% TESTING
#import bnlearn
#import pandas as pd

#G=bnlearn.import_DAG('asia', CPD=True)
#current_dag=G['model']
#df = bnlearn.sampling(G, n=10)
#generator = batch_generator(data=df, step_length=50)
#training_batch = next(generator)
#training_batch.shape
#batch = pd.DataFrame(training_batch[0, :, :], columns=df.columns, dtype=float)
#batch.dtypes

#for i in range(training_batch.shape[0]):
#    print(i)

#import matplotlib.pyplot as plt
#batch = 56   # First sequence in the batch.
#feature = 3  # First signal from the 20 input-signals.
#seq = training_batch[batch, :, feature]
#seq_1 = training_batch[batch, :, :]
#seq_1.shape
#plt.plot(seq)

#df = _sorter(df, data_order='ascending', distance_measure='cityblock')