#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:44:27 2020

@author: harit
"""

# Unzipping dataset in local directory
#from zipfile import ZipFile
#local_zip = 'creditcardfraud.zip'
#zip_ref = ZipFile(local_zip,'r')
#zip_ref.extractall()

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
frauds = mapping[w]
frauds = sc.inverse_transform(frauds)

# Going from Unsupervised learning to superwised learning

# creating frature matrix
customers = dataset.drop('Amount',axis=1)
customers = customers.iloc[:,:].values

# statndard scaling of feature matrix
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# creating dependent variable
is_fraud = np.zeros(len(customers))
for i in range(len(customers)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] =1

# creating ANN model
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 5)


# predicting the probabilities of frauds 
y_pred = classifier.predict(customers)
y_p = pd.DataFrame(data = y_pred , index =None , columns = 'Probability of Fraud')
output = pd.concat([dataset ,y_p],axis =1)