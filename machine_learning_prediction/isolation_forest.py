# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from numpy import quantile, random, where
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 20)


## 1-Dimensional Demo
random.seed(3)
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(20, 5))
plt.scatter(X[:, 0], X[:, 1], marker="o", c=_, s=25, edgecolor="k")
model = IsolationForest(n_estimators=100, contamination=.03)
predictions = model.fit_predict(X)
outlier_index = where(predictions==-1)
values = X[outlier_index]
plt.scatter(X[:,0], X[:,1])
plt.scatter(values[:,0], values[:,1], color='y')
plt.show()


## DF demo
df = pd.read_csv("~/Code/data/SpotifyDataset160k.csv", na_values=[" ", ""])
X = df.select_dtypes([int, float]).loc[::100,:]
model = IsolationForest(n_estimators=100, contamination=.003)
predictions = model.fit_predict(X)
outlier_index = where(predictions==-1) # Beware, this functions returns positions, not index values!
values = df.iloc[outlier_index[0]]
# Inspect values with variable explorer
values.count() # No info which columns arethe discriminative. So the pattern must be dected through intuition


## DF-Demo 2-Dimensional
df = pd.read_csv("~/Code/data/SpotifyDataset160k.csv", na_values=[" ", ""])
X = df.loc[::100, ['danceability', 'duration_ms']]
model = IsolationForest(n_estimators=100, contamination=.03)
predictions = model.fit_predict(X)
outlier_index = where(predictions==-1) # Beware, this functions returns positions, not index values!
values = X.iloc[outlier_index[0]]

# run next two lines together! Otherwise Sypder will not render it in one graph
sns.scatterplot(x=X['duration_ms'], y=X['danceability'])
sns.scatterplot(x=values['duration_ms'], y=values['danceability'], color='y')

