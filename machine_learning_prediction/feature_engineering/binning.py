# Familiar imports
import numpy as np
import pandas as pd; pd.set_option('mode.chained_assignment','raise');

# For training random forest model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the training data
df = pd.read_csv("~/Code/data/california_housing.csv", na_values=["", " "])

# Feature Binning
#
# It may be helpful to bin features in certain groups and reduce the cardinality
# This is helpful if a certain range can be grouped to a certain feature

# Easiest way is using cut
df.groupby(pd.cut(df.housing_median_age, 10).values).mean() # This groups according to a column
df.groupby(pd.cut(df.index, 10)).mean() # This just groups into bins through a incrementing index. If no incrementing index is present one must be created

# Or using KBinsDiscretizer
from sklearn.preprocessing import KBinsDiscretizer
kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform') # also encode='onehot' works. Then we must to .toarray later and reshape accordingly
column = df.housing_median_age.values.reshape(-1,1) # column must be in appropriate dimension
transformed_column = kbins.fit_transform(column)
transformed_column = transformed_column.reshape(1,-1)[0] # we must AGAIN reshape ... no way to handle this nicer???
df.join(pd.Series(transformed_column, name="housing_median_age_bin")) # rejoin with dataframe


# Feature Clustering
# 
# Different to pure binning the clusters are created by a measure
# In the classic form this is a measure of distance in an n-dimensional space. (n>=1)
#
# Following an example with 5 bins in a 4 dimensional space (aka 4 features)
#### The features should be selected according to some domain logic, and optionally graphically validated

from sklearn.cluster import KMeans

features = ["total_rooms", "total_bedrooms", "median_house_value", "households"] # these features are selected, cause they all describe a similar topic
# alternatively a set can be chosen which has no shared topic, but where an interacation is assumed

# for clustering no NaNs are allowed
df = df.bfill().ffill()

# For clustering features must always be standardized. Optionally also log may be helpful if gaps are extremly large
df_scaled = df.loc[:, features]
df_scaled = (df_scaled - df_scaled.mean(axis=0)) / df_scaled.std(axis=0)

kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
df["Cluster"] = kmeans.fit_predict(df_scaled)
df

import seaborn as sns
import matplotlib.pyplot as plt

df["Cluster"] = df.Cluster.astype("category")
for feature in features:
    sns.relplot(
        x=feature, y="median_house_value", hue="Cluster", data=df
    )
plt.show() 

## Also a single var may be clustered
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
df["Cluster_Target"] = kmeans.fit_predict(df.median_house_value.values.reshape(-1,1))

sns.relplot(
    x=df.index, y="median_house_value", hue="Cluster_Target", data=df
)
plt.show() 
