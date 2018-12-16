#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:16:44 2018

@author: alexandrugris
"""

import pandas as pd
import numpy as np

def next_idx():
    
    if 'iter' in dir(next_idx):
        return next_idx.iter.__next__()
    else:
        def fn(first):
            while True:
                yield first
                first += 1
                
        next_idx.iter = fn(0)
        return next_idx()
        

df = pd.DataFrame(columns=['Team 1', 'Team 2', 'Result'])

teams = ['A', 'B', 'C', 'D', 'E', 'F' , 'G']

df.loc[next_idx()] =  ['A', 'B', 1]
df.loc[next_idx()] =  ['A', 'C', 1]
df.loc[next_idx()] =  ['B', 'D', 1]
df.loc[next_idx()] =  ['B', 'E', 1]
df.loc[next_idx()] =  ['C', 'E', 1]
df.loc[next_idx()] =  ['C', 'D', 1]
df.loc[next_idx()] =  ['D', 'F', 1]
df.loc[next_idx()] =  ['E', 'F', 1]
df.loc[next_idx()] =  ['G', 'D', 1]
df.loc[next_idx()] =  ['G', 'E', 1]
df.loc[next_idx()] =  ['G', 'F', 1]


params = pd.DataFrame( [0.5] * len(teams), index = teams, columns = ['Value'] )
    
def f(params):
    
    params = pd.DataFrame(params, index=teams)

    df['Home'] = np.array(params.loc[df['Team 1']])
    df['Away'] = np.array(params.loc[df['Team 2']])

    # probability func P(Team 1 > Team 2)
    df['Rank'] = df['Home'] / (df['Away'] + df['Home'] + 1e-5)

    df['Log'] = np.log(np.array(df['Rank'] * df['Result'] + (1.0 - df['Rank']) * (1.0 - df['Result']), dtype=float))
    
    penalty = np.array(params)
    penalty -= penalty.mean() # tries to get everything closer to the mean; this can be commented out
    
    # remove the np.sqrt ... to remove the penalty for large coefficients
    # if you remove that, the teams that have not been beaten will have a max of 5 score 
    return -df['Log'].sum() #+ 0.1 * np.sqrt(np.sum(penalty * penalty))

from scipy.optimize import minimize, Bounds

x = minimize(f, np.array(params), bounds=Bounds([1e-5] * len(teams), [1] * len(teams)))

if x.success:
   probs = pd.DataFrame(x.x, index=teams)

def prob(t1, t2):
    return probs.loc[t1] / (probs.loc[t1] + probs.loc[t2])

prob('G', 'A')
