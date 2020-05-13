#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:54:17 2020

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

# importing libraries and keras packages
from keras.models import Sequential
from keras.layers import Dense , Dropout

# Initializing the ann
classifier = Sequential()

# adding layers to classifier
classifier.add(Dense(units = 16 , activation = 'relu' , input_dim = 30))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 32 , activation= 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1 , activation='sigmoid'))

# compiling ann 
classifier.compile(optimizer= 'adam' , loss = 'binary_crossentropy' ,metrics = ['accuracy'])

# Fitting the dataset to ann classifier
classifier.fit(x_train , y_train, batch_size =32 ,epochs = 10)

#predicting on test set
y_pred = classifier.predict(x_test)
y_pred = y_pred>0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#calculating accuracy
score = classifier.predict(x_test)
print (classifier.summary())

new_X = [ np.argmax(item) for item in score ]
y_test2 = [ np.argmax(item) for item in y_test]

# Calculating categorical accuracy taking label having highest probability
accuracy = [ (x==y) for x,y in zip(new_X,y_test2) ]
print(" Accuracy on Test set : " , np.mean(accuracy))
