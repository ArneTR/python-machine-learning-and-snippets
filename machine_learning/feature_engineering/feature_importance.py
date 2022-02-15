# Familiar imports
import numpy as np
import pandas as pd

# For training random forest model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import mutual_info_regression # use this for continuous vars
from sklearn.feature_selection import mutual_info_classif # use this for categorical vars


## Info ##
# Mutual information does a non-parametric regression. Usually a lasso or a splined regression
# Therefore this information can be very helpful to select features beforehand
# But it's most use is probably to NOT kick out features by accident that this algorithm shows high values for.


## IMPORTANT: 
# - Categorical variables must be encoded
# - If you have few rows in your data the mutual_info_score will be unusable. 
#   A general rule of thumb 10:1 for every variable ... meaning if you have 9 covariates you need 90 rows

## Continuous Target Variable
df_rich = pd.read_csv("~/Code/data/house_prices_kaggle/train.csv", index_col="Id", na_values=["", " "])
df_rich = df_rich.drop(df_rich.columns[(df_rich.isna().any())], axis="columns")
X = df_rich.drop("SalePrice", axis="columns").select_dtypes([int,float])
y = df_rich.SalePrice
mi_scores = mutual_info_regression(X,y)
pd.Series(mi_scores,index=X.columns).sort_values()



## Binary / categorical target variable
df_few = pd.read_csv("~/Code/data/Social_Network_Ads.csv")
df_few.Gender, _ = df_few.Gender.factorize()
df_few = df_few.loc[0:10]
X = df_few.drop("Purchased", axis="columns")
y = df_few.Purchased
mi_scores = mutual_info_classif(X,y)
pd.Series(mi_scores,index=X.columns)

df_few.groupby("Age").Purchased.value_counts(normalize=True)