#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 09:47:34 2018

@author: alexandrugris
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def import_data(*teams):
    ret = {}
    
    for team in teams:
        
        df = pd.read_csv(team + ".csv", )
        df = df.set_index(df.columns[0])
        df.fillna(0, inplace=True)
        
        ret[team] = df / 100 # modify to probabilities
        
    return ret
    

data = import_data('brighton', 'totenham')

### these points to be discussed
fp = {
      # first 3, exclusive 
       '#1 To Score 1 goal' :  [2, 0],
       '#2 To Score 2 goals' : [2, 0],
       '#3 To Score 3 goals' : [2, 0],
       
       '#4 Make an assist' : [1,1],
       
       '#5 Play >60 minutes' : [1, 2],
       '#6 Play full match': [1, 2], # will add in total 2 points, 
       
       '#7 Get a yellow card' : [-1, 3],
       '#8 Get a red card (edited' : [-2, 4],
       '#9 keeper_save( 3 saves) for GK': [3, 5]
      }

fp = pd.DataFrame.from_dict(fp, orient='index')


fp[0] = fp[0].astype('float')


def xfp(team, fp):
    return (team * fp).sum(axis='columns')

# to compute the expected fp from expected values, use xfp(data['brighton'], fp)
    
# below, we use simulation
# values are approx similar for xfp, but variation is quite large

def simulate_team(team, fp):
    
    # generate 6 random numbers per player,
    probs = np.random.rand(team.shape[0], 6)
    events = team > probs[0:team.shape[0], fp]
    
    # make comparison with probabilities to see if events occured
    # first event is the number of goals, so uses the same random number
    # same for duration of play
    return events
    
def run_simulation(team, cnt):
    
    sim_df = pd.DataFrame(index=team.index)
    
    for i in range(0, cnt):
        s = xfp(simulate_team(team, fp[1]), fp[0])
        sim_df[i] = s 
        
    return sim_df
   
def run_simulation_all(data):
    
    stats_sym = {}
    
    for k, team in data.items():
        x = run_simulation(team, 1000)
        
        # comment the lines below to remove statistics
        # but they don't contribute much to the algorithm slowness
        
        s = x.T.describe()
        s.loc['sum'] = x.sum(axis='columns').T
        
        stats_sym[k] = (x, s)
        
    return stats_sym


def pvp_xfp(sym, team1, name1, team2, name2):
    
    p1 = sym[team1][0].loc[name1]
    p2 = sym[team2][0].loc[name2]
    
    t1 = np.sum(p1 > p2) / p1.size
    tx = np.sum(p1 == p2) / p1.size
    t2 = np.sum(p1 < p2) / p1.size
        
    return (1 / t1, 1 / tx , 1 / t2)
        

sym = run_simulation_all(data)
pvp_xfp(sym, 'brighton', 'Bong', 'totenham', 'Michel Vorm')

# some data to play around
p1 = sym['brighton'][0].loc['Bong']
p2 = sym['totenham'][0].loc['Michel Vorm']

p1.hist(bins=100)
p2.hist(bins=100)

# logarithm does not help, data is too rare
#p1 = np.log(p1).replace([np.inf, -np.inf], np.nan).dropna()
#p2 = np.log(p2).replace([np.inf, -np.inf], np.nan).dropna()

x = sym['totenham'][0]
columns = range(int(x.min().min()), int(x.max().max() + 1e-5))

df = pd.DataFrame(columns=columns, index=x.index)


for name, row in x.iterrows():
    print(row.value_counts())
    df.loc[name] = row.value_counts(normalize=True)

######
##### team vs team

brighton = distributions(sym['brighton'][0])
totenham = distributions(sym['totenham'][0])

def to_dict(team: pd.DataFrame) -> dict:
    """ Exports to python dictionary a dataframe containing team results"""
    return team.T.to_dict()

def from_dict(team: dict) -> pd.DataFrame:
    """ Restores a dataframe team from a python dictionary"""
    return pd.DataFrame(team).T

def team_distrib(team: pd.DataFrame) -> pd.DataFrame:
    """ Computes the distribution of xfp for a team """
    
    team = team.cumsum(axis='columns')
    
    team_points = []
    
    for i in range(0, 1000):        
        rnd = np.random.rand(team.shape[0]).reshape(team.shape[0],1).repeat(team.shape[1], axis=1)
        evts = team[team < rnd].T.idxmax()
        team_points.append(evts.sum())
    
    return distributions(pd.DataFrame(team_points).T)

def team_vs_team_hda(t1: pd.DataFrame, t2: pd.DataFrame) -> pd.DataFrame:
    """ Basic HDA for team vs team. """
    
    c1 = t1.columns.sort_values()
    c2 = t2.columns.sort_values()
    
    h = 0
    d = 0
    a = 0
    
    for i in c1:
        for j in c2:
            v = float(t1[i] * t2[j])
            if i < j:
                 h += v
            elif i == j:
                d += v
            else:
                a += v
                
    return (1/h, 1/d, 1/a)   
                
hda_team_vs_team = []

for i in range(0, 10):
    hda_team_vs_team.append(team_vs_team_hda(team_distrib(brighton), team_distrib(totenham)))

hda_df_t = pd.DataFrame(hda_team_vs_team)
print(hda_df_t.sem(axis='rows'))

