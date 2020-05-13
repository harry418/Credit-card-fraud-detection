#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:14:36 2020

@author: harit
"""

# Import necessary library
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# spilliting training or test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# applying kernel PCA on training set
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components = 2 , kernel ='rbf')
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

# predicting on test set
y_pred = classifier.predict(x_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

new_X = [ np.argmax(item) for item in y_Pred ]
y_test2 = [ np.argmax(item) for item in y_test]

# Calculating categorical accuracy taking label having highest probability
accuracy = [ (x==y) for x,y in zip(new_X,y_test2) ]
print(" Accuracy on Test set : " , np.mean(accuracy))
