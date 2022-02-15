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



# Scale / Transform data
#
# This is a very delicate topic, as Scalings may make it harder to interpret data in the end.
# Usually the interrelation of the values in a columns are preserved (ex. value 4 is X times larger as value 9, same relation is valid for 3 and 7).
# But the relationship may not be the same size anymore. Ex. it is not 3-times larger anymore, but a different, scaled, factor.
#
# The other problem is, if you have zero values in your data. In that case you should not use StandardScaler, as this might lead to wrong results.
# The zero has an implicing meaning ,that would be overwritten by the STandardScaler

if features.isin([0]).any().any():
    raise ValueError("Dataframe contains Zeros! Please be vary when using StandardScaler!")

scan_cols = features.select_dtypes([float, int])

from sklearn.preprocessing import StandardScaler, RobustScaler
s = StandardScaler() # Use if data is already approx Normal-distributed and if 0 has no meaning and if no big outliers are present
s = RobustScaler() # Use if data is already approx Normal-distributed and if 0 has no meaning and if outliers are present
# - LogTransform() # Use this if the data is highly skewed or very condensed in some parts of it's range. LogTransform will stretch condensed parts and 
#   condense stretched parts so the model can work better with the data
## # If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log


# You can also use QuantileTransformers as a test ... but output is hard to interpret: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

features[scan_cols.columns.tolist()] = s.fit_transform(scan_cols)


# Then later call s.transform(test_data) on the test set to apply pre-calculated transform

s.inverse_transform(features[scan_cols.columns.tolist()]) # on the data later to get original values back


## Create new Features

# Method 1: Domain knowledge.
# - Some columns make sense to substract?
# - See non-linear relationship?
# - Create Sums / Averages for specific category
# - Create new function? For example angular velocity from linear velocity through 2*PI*1/T
# - Fourier-Transform? Integrate? etc.
# - Bin Columns? For instance group in 10-bins? or find logical clusters? -> See separate file
# - Process String Columns? For instance only use last names in columns? -> See separate file


## Example: Create Mean for specific group / category and apply as new column for EVERY row (not only as groupby().agg() )
customer = pd.read_csv("~/Code/data/customer.csv")
customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

customer[["Customer", "State", "Response", "SalesChannel", "Income", "AverageIncome"]] # see for the last ones, all california, all same AvgIncome




# Method 2: auto create features by using. Later then check if we have correlation

def runcorr(i, features):

    num_cols = features.drop(columns='target').select_dtypes([float, int])

    for j in num_cols:
            if i != j:
                features["%s - %s" % (i,j)] = features[i]-features[j]
                features["%s + %s" % (i,j)] = features[i]+features[j]
                features["%s * %s" % (i,j)] = features[i]*features[j]
                features["%s * %s" % (i,i)] = features[i]*features[i] # Quadratic relationship. Maybe even add polynomial or log ????

    return features

# since the enhanched matrices explode very quickly in datasize, we run them one by one
for i in range(2,11):
    print("cont%s" % i)
    train_features = runcorr("cont%s" % i, train_features)
    corr = train_features.corr()
    print(corr.target[corr.target > 0.5])
    del corr
    del train_features



## Model an interaction effect:

df_new = pd.get_dummies(df.CATEGORICAL_FEATURE, prefix="Cat") # One-hot encode Categorical feature, adding a column prefix "Cat"
df_new = df_new.mul(df.CONTINUOUS_FEATURE, axis=0) # Multiply row-by-row
df = df.join(df_new) # Join the new features to the feature set

