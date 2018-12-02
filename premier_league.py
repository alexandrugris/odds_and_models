#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:38:10 2018

@author: alexandrugris
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://datahub.io/sports-data/english-premier-league#readme

data = pd.read_csv("season-1718_csv.csv") # todo: add index on team names!

rows = len(data)
teams = int(np.sqrt(rows)) + 1

# home Tottenham, away Brighton


home = data['FTHG']
away = data['FTAG']

avg_goals_scored = (home.mean() + away.mean()) / 2

def attack_defence(team):
    attack = (data['FTHG'].loc[data['HomeTeam'] == team].mean() + data['FTAG'].loc[data['AwayTeam'] == team].mean()) / (2 * avg_goals_scored)
    defence = (data['FTAG'].loc[data['HomeTeam'] == team].mean() + data['FTHG'].loc[data['AwayTeam'] == team].mean()) / (2 * avg_goals_scored)

    return (attack, defence)

def xg(home_, away_):

    h_attack, h_defence = attack_defence(home_)
    a_attack, a_defence = attack_defence(away_)
    
    xGH = h_attack * a_defence * home.mean()
    xGA = a_attack * h_defence * away.mean()
    
    return (xGH, xGA)

def cnt_goals(team):
    return data['FTHG'].loc[data['HomeTeam'] == team].sum() + data['FTAG'].loc[data['AwayTeam'] == team].sum()
     
# compute the expected goals for each team
xGH, xGA = xg('Brighton', 'Tottenham')

from scipy.stats import poisson

def hda(xGH, xGA):
    h = np.array([poisson.pmf(x, xGH) for x in range(0, 10)])
    a = np.array([poisson.pmf(x, xGA) for x in range(0, 10)])
    
    ret = [0, 0, 0]
    
    for i in range (0, 10):
        for j in range (0, 10):
            if i > j:
                ret[0] += h[i] * a[j]
            if i == j:
                ret[1] += h[i] * a[j]
            if i < j:
                ret[2] += h[i] * a[j]
                
    return 1 / np.array(ret)

def expected_goal_odds(row):
    
    hm = row['HomeTeam']
    aw = row['AwayTeam']
    hmg = row['FTHG']
    awg = row['FTAG']
    
    xGH, xGA = xg(hm, aw)
    [h, d, a] = hda(xGH, xGA)
    return (hm, aw, xGH, xGA, h, d, a, row['FTR'], hmg > awg, hmg == awg, hmg < awg)

expected_goals = data.apply(expected_goal_odds, axis=1, result_type='expand') 
expected_goals.columns=['HomeTeam', 'AwayTeam', 'xGH', 'xGA', 'HOdds', 'DOdds', 'AOdds', 'Result' ,'H', 'D', 'A']


def attack_defence_spread(row):
    
    hm = row['HomeTeam']
    aw = row['AwayTeam']
    hmg = row['FTHG']
    awg = row['FTAG']
    
    atk_h, def_h = attack_defence(hm)
    atk_a, def_a = attack_defence(aw)
    
    spread = hmg - awg
    
    return (hm, aw, atk_h, def_h, atk_a, def_a, spread)
    

atk_def_spread = data.apply(attack_defence_spread, axis=1, result_type='expand')
atk_def_spread.columns = ['Home', 'Away', 'AtkH', 'DefH', 'AtkA', 'DefA', 'Spread']

from sklearn.linear_model import LinearRegression

X_train = atk_def_spread[['AtkH', 'DefH', 'AtkA', 'DefA']]
Y_train = atk_def_spread[['Spread']]

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# check R^2
print(regressor.score(X_train, Y_train))

# check residuals
y_pred = regressor.predict(X_train)
e = Y_train  - y_pred
e.hist()

line = atk_def_spread.loc[(atk_def_spread['Home'] == 'Tottenham') & (atk_def_spread['Away'] == 'Brighton')]

spread_T_vs_B = regressor.predict(line[['AtkH', 'DefH', 'AtkA', 'DefA']])[0][0]

## compute prob of winning
from scipy.stats import norm as norm
h = 1 - norm.cdf(0.5, loc=spread_T_vs_B, scale=e.std())[0]
a = norm.cdf(-0.5, loc=spread_T_vs_B, scale=e.std())[0]
d = 1 - h - a

[1/h, 1/d, 1/a]


#####

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(expected_goals[['xGH', 'xGA', 'HOdds', 'DOdds', 'AOdds']], expected_goals[['Result']], test_size=0.10, random_state=42)

regressor = LogisticRegression(multi_class='multinomial', solver='lbfgs')
regressor.fit(X_train, y_train['Result'])

y_pred = regressor.predict(X_test)

new_odds = pd.DataFrame(1 / regressor.predict_proba(X_test), index=X_test.index)
new_odds.columns = ['L_A', 'L_D', 'L_H']
new_odds = new_odds[['L_H', 'L_D', 'L_A']]

odds_comparison = X_test.join(new_odds)#.join(expected_goals[['HOdds', 'DOdds', 'AOdds']])

plt.scatter(odds_comparison['DOdds'], odds_comparison['L_D'])


#####
# similarly we can compute for red cards, yellow cards, assistst
    
player1 = 5 / 32  # brighton -> number of goals / match
player2 = 30 / 37 # totenham -> number of goals / match

team_percentage_player1 = player1 * rows / (teams * cnt_goals('Brighton'))
team_percentage_player2 = player2 * rows / (teams * cnt_goals ('Tottenham'))

xGPlayer1 = team_percentage_player1 * xGH
xGPlayer2 = team_percentage_player2 * xGA


def goals(player):
    return [poisson.pmf(0, player), poisson.pmf(1, player), poisson.pmf(2, player), poisson.pmf(3, player)]

goals(xGPlayer1)
goals(xGPlayer2)
