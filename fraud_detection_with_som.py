#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:07:36 2020

@author: harit
"""

# Import necessary library
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
sc.fit_transform(X)

# Training the som
from minisom import MiniSom
som = MiniSom(x = 10 ,y = 10 ,input_len = 30 , learning_rate = 0.5 ,sigma = 1.0)
som.random_weights_init(X)
som.train_random(data = X ,num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# finding the frauds
mapping = som.win_map(X)
frauds = mapping[(8,8)]
frauds = sc.inverse_transform(frauds)