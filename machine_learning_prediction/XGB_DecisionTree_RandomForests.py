# Familiar imports
import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

data = pd.read_csv("~/Code/data/Social_Network_Ads.csv", na_values=["", " "])

print(data.describe())

print(data.head())

features = data.copy() # make working copy


features.isna().sum() # Data contains no NA values

# remove unneeded column
features.drop(columns="User ID", inplace=True)

# Transform
features.Gender = features.Gender.factorize()[0]


# Be careful with outliers.
# Removal MIGHT help if you suspect them to be irregular.
# But if outliers are highly discriminative they are very helpful in
# Tree based models, cause they can be easily separated
scan_cols = features.drop(columns='Purchased').select_dtypes([float, int])
for i,col in enumerate(scan_cols):
    index = features[col][(features[col] - features[col].mean()).abs() > 3*features[col].std()]
    print("%s scanned:" % col, index.shape)
    features.drop(index=index.index, inplace=True)


# Prebalancing
# This step is optional and should only be considered, if the target has many categories that
# are outbalanced. Say 200000x A + 1000x B + 1000x C.
# The tree might focus too much on A and neglect the other cases. This however may be bad for
# future classification if C gets more important in production
#
# Therefore it must be determined how often the classes should be present as target variable
# Important: Note that purging too many rows may make the tree bad cause of low sample  size
answer = input("\n\nImportant: Do you want to use tree balancing. Press y for yes or any key for no ...\n\n")
if(answer == "y"):
    print("Using balancing")
    data = data[data.Purchased == 0][::2].append(data[data.Purchased == 1])


y = features.pop("Purchased")


# When using DecisionTrees or RandomForest it is import to first decide if there is
# a regression problem or a classification problem
#
# Since we have two categories here: Clicked / NotClicked, we use Classification
#
# This means RandomForestClassifier instead of RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.2, random_state = 50)

# now we compare the performances blank
print("RandomForestClassifier Train Performance: ", RandomForestClassifier().fit(X_train, y_train).score(X_train, y_train))
print("DecisionTreeClassifier Train Performance: ", DecisionTreeClassifier().fit(X_train, y_train).score(X_train, y_train))
print("ExtraTreeClassifier Train Performance: ", ExtraTreeClassifier().fit(X_train, y_train).score(X_train, y_train))
print("XGBClassifier Train Performance: ", XGBClassifier(use_label_encoder=False).fit(X_train, y_train).score(X_train, y_train))
print("\n")
print("RandomForestClassifier Test Performance: ", RandomForestClassifier().fit(X_train, y_train).score(X_test, y_test))
print("DecisionTreeClassifier Test Performance: ", DecisionTreeClassifier().fit(X_train, y_train).score(X_test, y_test))
print("ExtraTreeClassifier Test Performance: ", ExtraTreeClassifier().fit(X_train, y_train).score(X_test, y_test))
print("XGBClassifier Test Performance: ", XGBClassifier(use_label_encoder=False).fit(X_train, y_train).score(X_test, y_test))


print("\n\nAs we can see the Trees usually overfit with the given standard params. According to the docs it is a good idea to limit the tree size starting with 3 and then going up, or setting min_leaf_size\n\n")



print("RandomForestClassifier Train Performance with max_depth=3: ", RandomForestClassifier(max_depth=3).fit(X_train, y_train).score(X_train, y_train))
print("DecisionTreeClassifier Train Performance with max_depth=3: ", DecisionTreeClassifier(max_depth=3).fit(X_train, y_train).score(X_train, y_train))
print("ExtraTreeClassifier Train Performance with max_depth=3: ", ExtraTreeClassifier(max_depth=3).fit(X_train, y_train).score(X_train, y_train))
print("\n")
print("RandomForestClassifier Test Performance with max_depth=3: ", RandomForestClassifier(max_depth=3).fit(X_train, y_train).score(X_test, y_test))
print("DecisionTreeClassifier Test Performance with max_depth=3: ", DecisionTreeClassifier(max_depth=3).fit(X_train, y_train).score(X_test, y_test))
print("ExtraTreeClassifier Test Performance with max_depth=3: ", ExtraTreeClassifier(max_depth=3).fit(X_train, y_train).score(X_test, y_test))
print("XGBClassifier Test Performance: ", XGBClassifier(n_estimators = 100, max_depth=3, use_label_encoder=False).fit(X_train, y_train).score(X_test, y_test))
