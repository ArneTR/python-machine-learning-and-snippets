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


# check outliers - must be removed if wanted
scan_cols = train.drop(columns='target').select_dtypes([float, int])

# Remove outliers
for i,col in enumerate(scan_cols):
    index = train_features[col][(train_features[col] - train_features[col].mean()).abs() > 3*train_features[col].std()]
    print("%s scanned:" % col, index.shape)
    train_features.drop(index=index.index, inplace=True)


