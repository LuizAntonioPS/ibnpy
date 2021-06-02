# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:17:46 2021

@author: LuizPereira
"""

import numpy as np
import pandas as pd

def _append_score(df):
    # score deverá explicar a diferença entre as distribuições explicadas pelas duas linhas. A ideia da ordenação é que os algoritmos recebem dados que expliquem distribuições similares (random) ou divergentes (similar e dissimilar) para que seja avaliado em ambos os contextos, onde, em um deles, as dados são sempre gerados por distribuições diferentes
    # quando for ordenação não randomica, seta o score da instância base como MAX_VALUE ou MIN_VALUE para não entrar na ordenação
    base_row = df.iloc[0]
    rows = df.iloc[1:]
    scores = np.random.randn(len(df))
    df['score'] = pd.Series(scores)
    return df

def _sorter(df, data_order='random'):
    #df = _append_score(df)
    if data_order != 'random':
        ascending = data_order == 'ascending'
        df = df.sort_values(by=['score'], ascending=ascending)
    return df

## TESTING
#import bnlearn
#import pandas as pd
#G=bnlearn.import_DAG('asia', CPD=True)
#df = bnlearn.sampling(G, n=10)
#df = _sorter(df, data_order='desceding')  


    
def batch_generator(data,
                    data_order='random',
                    features=None,
                    classes=None,
                    batch_size=None,
                    step_length=100
                    ):
    """
    Generator function for creating batches of training-data.

    """
    
    data = _sorter(df=data, data_order=data_order)
    
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
            idx = np.random.randint(len(x) - step_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x[idx:idx+step_length]
            if not classes is None:
                y_batch[i] = y[idx:idx+step_length]
        
        if not classes is None:
            yield (x_batch, y_batch)
        else:
            yield x_batch
            
            
            
## TESTING
#import bnlearn
#import pandas as pd

#G=bnlearn.import_DAG('alarm', CPD=True)
#current_dag=G['model']
#df = bnlearn.sampling(G, n=10000)
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
