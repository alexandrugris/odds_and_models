#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 12:09:18 2018

@author: alexandrugris
"""

import numpy as np
import pandas as pd



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
    return (team * fp[0]).sum(axis='columns')

# no of clusters in which to split the players based on their performance
clusters = 5

# budget I have at my disposal
budget = 100

# number of players I have to have in my fantasy team
max_players = 5

#player_count = 24
#player_names = [chr(ord('A') + x) for x in range(0, player_count)]
#player_scores = 2 ** np.random.normal(size=len(player_names))
#player_scores = player_scores / np.mean(player_scores)
#players = pd.DataFrame(player_scores, index=player_names)


players = pd.DataFrame(xfp(data['brighton'], fp).append(xfp(data['totenham'], fp)))
players.sort_values([0]).plot()

from sklearn.cluster import KMeans

clusters = KMeans(n_clusters = clusters).fit(players)

print(clusters.cluster_centers_)


counts = [clusters.labels_[clusters.labels_ == x].size for x in range(0, clusters.cluster_centers_.size)]

def valid_combinations(max_players, counts, start_with=0):
        
    if max_players <= 0:
        return []
    
    if max_players == 1:
        for i in range(start_with, len(counts)):
            if counts[i] > 0:
                yield [i]
        
    for i in range(start_with, len(counts)):
        if counts[i] > 0:
            cnts = [counts[x] if x != i else counts[x] - 1  for x in range(len(counts))]
            for x in valid_combinations(max_players - 1, cnts, i):
                yield [i] + x
             
def fp(x):
    return np.sum(clusters.cluster_centers_[x])
                
f = sorted([(x, fp(x)) for x in valid_combinations(max_players, counts)], key=lambda x : x[1], reverse=True)

def counts_f(f):
    f_ = [x for x, _ in f]
    return np.array([[xf.count(i) for i in range(0, clusters.cluster_centers_.size)] for xf in f_])

# optimize budget for top percent of the possible lineups

df_avg = counts_f(f[0: int(2 * len(f)/ 3)])
df_max = counts_f(f[0: int(len(f)/5)])

from scipy.optimize import minimize

def team_prices(params):
    f = np.sum(df_max * params)
    return -f

def team_prices_avg(params):
    f = df_avg.dot(params) - np.array([100] * len(df_avg))
    err = -np.sqrt(np.sum(f * f)) / len(f) + budget * 0.05 # 5 percent of budget
    return err

def min_price(params):
    return min(params) - budget * 0.05

def max_price(params):
    """ makes sure max price is selectable """
    
    s = max(params)
    cnt = 1
    pc = sorted([[params[i], counts[i] if params[i] < s else counts[i]-1] for i in range(len(params))], key=lambda x : x[0], reverse=False)

    for i in range(0, len(pc)):
        while cnt < max_players and pc[i][1] > 0:
            s += pc[i][0]
            pc[i][1] -= 1
            cnt += 1
        
    return budget * 0.95 - s

clstrs = clusters.cluster_centers_.size
x = minimize(team_prices, np.array([budget / clstrs] * clstrs), method='COBYLA', constraints = (
        { 'type': 'ineq', 'fun': team_prices_avg },
        { 'type': 'ineq', 'fun': min_price },
        { 'type': 'ineq', 'fun': max_price }))

results = pd.DataFrame([x.x, [x[0] for x in clusters.cluster_centers_], counts]).T
results.columns = ['Price Millions', 'XFP for cluster', 'Count of players']

### possible selections for playing
selections = pd.DataFrame([x for x in counts_f(f)])
selections['XFP']   = selections.dot(clusters.cluster_centers_.reshape((clstrs,1)))
selections['Price'] = selections[[i for i in range(0,clstrs)]].dot(x.x)

# individual players
players.columns = ['XFP']
players['Cluster'] = clusters.labels_
players['Cluster Price'] = x.x[clusters.labels_]

def price_in_cluster(xfp, cluster_xfp_values, cluster_price):
    mn = min(cluster_xfp_values)
    mx = max(cluster_xfp_values)
    
    if mn == mx:
        mn = xfp - 0.1 * xfp
        mx = xfp + 0.1 * xfp
    
    pos = (xfp - mn) / (mx - mn)
    return np.random.normal(cluster_price * (0.85 * (1-pos) + 1.05 * (pos)), cluster_price * 0.05)

def cluster_xfp_values(cluster):
    return players.loc[clusters.labels_ == cluster]['XFP']
    

players['Final Price'] = [price_in_cluster(r['XFP'], cluster_xfp_values(r['Cluster']), r['Cluster Price']) for i, r in players.iterrows() ]

