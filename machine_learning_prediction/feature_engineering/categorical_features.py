# Familiar imports
import numpy as np
import pandas as pd; pd.set_option('mode.chained_assignment','raise');

# For training random forest model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the training data
data = pd.read_csv("~/Code/data/house_prices_kaggle/train.csv", index_col="Id", na_values=["", " "])

features = data.copy()


# Encode Categorical Data
#
# The easiest way to encode data is by calling .factorize. However, this mapping
# will not be preserved on later calls to the function. There here using a Encoder class
# is mandatory
#
# There are many ways to take, but should follow this order:
# - OrdinalEncoder (Same as LabelEncoder logically. But LabelEncoder is only for 1-D Data)
#     - This should ONLY be used on ordered data, as models may interpret the ascending nature of the ordering
# - OneHotEncoder - Generally good. But increases complexity very strong. 16 categories per Column is a sane upper limit!


scan_cols = features.select_dtypes([object])

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Method 1 - Ordinal Encoding
## This method may be problematic, cause it creates an order that is not present in the initial data
o = OrdinalEncoder() 
features[scan_cols.columns.tolist()] = o.fit_transform(scan_cols)
## Alternative: df.COLUMN.factorize()


# Method 2 - One Hot Encoding
## This method may be problematic, cause it creates very many features
o = OneHotEncoder(handle_unknown="ignore", sparse=False) 
encoded_cols = o.fit_transform(scan_cols)
encoded_df = pd.DataFrame(encoded_cols, index=features.index)
features.join(encoded_df) # Join uses by default the DataFrame Index as join condition for the left-join
features.drop(scan_cols.columns.tolist(), inplace=True)
#### IMPORTANT: Way better method: pd.get_dummies(df.COLUMN)



# Method 3 - Target Encoding / Leave-one-out Encoding
## This method connects the target variable with the categorical data

## TODO

# Create features from categorical features


## Sum of all categorical features
accidents = pd.read_csv("~/Code/data/accidents.csv")
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

accidents[roadway_features + ["RoadwayFeatures"]].head(10)# => This will create a sum of all True features for a given row


## Count of values in categorical feature
# Notice the use of .gt as modifier!

## Important!!!! Counts are especially helpful for tree models, since these models don't have a natural way of aggregating information across many features at once.

concrete = pd.read_csv("~/Code/data/concrete.csv")
components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1) # sum on ROW level, not column level

concrete[components + ["Components"]].head(10)


## Frequency-Encoding
# Count how ofen the relative occurence of the category is.
customer = pd.read_csv("~/Code/data/customer.csv")
customer["State_Ratio"] = customer.groupby("State")["State"].transform("count") / customer.shape[0] 
customer