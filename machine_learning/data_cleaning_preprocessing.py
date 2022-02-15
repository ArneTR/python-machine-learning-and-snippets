# Familiar imports
import numpy as np
import pandas as pd

# For training random forest model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the training data
df = pd.read_csv("~/Code/data/house_prices_kaggle/train.csv", index_col="Id", na_values=["", " "])

features = df.copy()

# check for unclear empty values
(df == "").any().any()
## or
(df == "empty").any().any()

# check outliers - must be removed if wanted
scan_cols = features.drop(columns='SalePrice').select_dtypes([float, int])


# Remove outliers
for i,col in enumerate(scan_cols):
    index = features[col][(features[col] - features[col].mean()).abs() > 3*features[col].std()]
    print("%s scanned:" % col, index.shape)
    features.drop(index=index.index, inplace=True)

# Remove / impute NA
for i,col in enumerate(features.select_dtypes([float, int])):
    print("Column: ", col, " has NA? ", features[col].isna().any(), ": ", features[col].isna().sum())
    # CHOOSE
    # features = features.dropna(subset=[col], axis="index") #V1
    # features[col]= features[col].fillna(features[col].mean())  # V2

for i,col in enumerate(features.select_dtypes([object])):
    print("Column: ", col, " has NA? ", features[col].isna().any(), ": ", features[col].isna().sum())
    # CHOOSE
    # features[col]= features[col].fillna(features[col].mode())  # V3 only for object cols


# WARNING: All of these steps can and should be done BEFORE calling train_test_split and must also be done on
# the final production set as well.
# Sometimes it is needed to carry on the behaviour of the training set. Especially if we only get feeded one
# row of data at a time and have to make a prediction on that.
# Here calling .mean() is useless, as our new data-set is only one sample. We therefore would usually fill in the
# old mean(), which we may not have saved. Therefore a Imputer Class, that remembers these values is helpful.
# Example:
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent") # can also be mean, median etc.
to_impute_columns = features.select_dtypes([object])
features[to_impute_columns.columns.tolist()] = imp.fit_transform(to_impute_columns)
# Now you can call imp.transform again with the set imputation scheme, and it will reapply the pre-calculated most_frequent / mean / median etc.


