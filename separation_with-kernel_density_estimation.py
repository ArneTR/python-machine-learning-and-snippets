#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 22:36:42 2022

@author: light
"""


from sklearn.neighbors import KernelDensity
import numpy as np
import seaborn as sns

np.random.seed(135)
def prepData(N):
    X = []
    for i in range(n):
        A = i/1000 + np.random.uniform(-4, 3)
        R = np.random.uniform(-5, 10)
        if(R >= 8.6):
            R = R + 10
        elif(R < (-4.6)):
            R = R +(-9)        
        X.append([A + R])   
    return np.array(X)
 
n = 500
X = prepData(n)


x_ax = range(n)


kern_dens = KernelDensity()
kern_dens.fit(X)
 
scores = kern_dens.score_samples(X)
threshold = np.quantile(scores, .1)
print(threshold)

idx = np.where(scores <= threshold)
values = X[idx]

sns.scatterplot(x=x_ax, y=X[:, 0]) 
sns.scatterplot(x=idx[0],y=values[:,0], color='r') # Kernel density makes a dynamic border. Sometimes it is lower, sometimes higher


y_low = np.where(X > (X.mean() + 2*X.std()) )
y_high = np.where(X < (X.mean() - 2*X.std()))

y = np.append(y_low[0], y_high[0])

sns.scatterplot(x=x_ax, y=X[:, 0]) 
sns.scatterplot(x=y,y=X[y,0], color='r') # This is a fixed border
