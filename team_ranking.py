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

df.loc[next_idx()] =  ['A', 'B', 1]
df.loc[next_idx()] =  ['A', 'C', 1]
df.loc[next_idx()] =  ['B', 'C', 0]
df.loc[next_idx()] =  ['B', 'D', 0]
df.loc[next_idx()] =  ['A', 'D', 0]
df.loc[next_idx()] =  ['C', 'B', 0]
df.loc[next_idx()] =  ['A', 'D', 1]
df.loc[next_idx()] =  ['D', 'A', 0]
df.loc[next_idx()] =  ['E', 'D', 1]
df.loc[next_idx()] =  ['A', 'E', 1]


params = pd.DataFrame.from_dict({
        'A': 0.5,
        'B' : 0.5,
        'C' : 0.5,
        'D' : 0.5,
        'E' : 0.5,
        }, orient='index', columns=['Value'])
    
def f(params):
    
    params = pd.DataFrame(params, index=['A', 'B', 'C', 'D', 'E'])

    df['Home'] = np.array(params.loc[df['Team 1']])
    df['Away'] = np.array(params.loc[df['Team 2']])

    # probability func P(Team 1 > Team 2)
    df['Rank'] = df['Home'] / (df['Away'] + df['Home'] + 1e-5)

    df['Log'] = np.log(np.array(df['Rank'] * df['Result'] + (1.0 - df['Rank']) * (1.0 - df['Result']), dtype=float))
    
    return -df['Log'].sum()

from scipy.optimize import minimize, Bounds

x = minimize(f, np.array(params), bounds=Bounds([1e-5, 1e-5, 1e-5, 1e-5, 1e-5], [5, 5, 5, 5, 5]))

if x.success:
   probs = pd.DataFrame(x.x, index=['A', 'B', 'C', 'D', 'E'])

def prob(t1, t2):
    return probs.loc[t1] / (probs.loc[t1] + probs.loc[t2])

prob('E', 'A')
