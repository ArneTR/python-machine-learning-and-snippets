# Familiar imports
import numpy as np
import pandas as pd

# For training random forest model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the training data
data = pd.read_csv("~/Code/data/house_prices_kaggle/train.csv", index_col="Id", na_values=["", " "])

features = data.copy()

# Method 1: Domain knowledge.
# - Some columns make sense to substract?
# - See non-linear relationship?
# - Bin Columns? For instance group in 10-bins? or find logical clusters?
# - Process String Columns? For instance only use last names in columns?


# Method 2: auto create features by using. Later then check if we have correlation

def runcorr(i, features):

    num_cols = features.drop(columns='target').select_dtypes([float, int])
    cat_cols = features.select_dtypes(object)

    for var in cat_cols:
        features[var] = features[var].factorize()[0]

    for j in num_cols:
            if i != j:
                features["%s - %s" % (i,j)] = features[i]-features[j]
                features["%s + %s" % (i,j)] = features[i]+features[j]
                features["%s * %s" % (i,j)] = features[i]*features[j]
                features["%s * %s" % (i,i)] = features[i]*features[i] # Quadratic relationship. Maybe even add polynomial or log ????

                for k in cat_cols:
                    features["%s * %s" % (i,k)] = features[i]*features[k]
                    features["%s * %s" % (j,k)] = features[j]*features[k]

                    features["(%s - %s) * %s" % (i,j,k)] = (features[i]-features[j])*features[k]
                    features["(%s + %s) * %s" % (i,j,k)] = (features[i]+features[j])*features[k]
                    features["(%s * %s) * %s" % (i,j,k)] = (features[i]*features[j])*features[k]
    return features

# since the enhanched matrices explode very quickly in datasize, we run them one by one
for i in range(2,11):
    print("cont%s" % i)
    train_features = runcorr("cont%s" % i, train_features)
    corr = train_features.corr()
    print(corr.target[corr.target > 0.5])
    del corr
    del train_features



