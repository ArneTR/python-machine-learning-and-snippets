#!/usr/bin/env python
# coding: utf-8

# # Step 1: Import helpful libraries
# 

# In[17]:


# Familiar imports
import numpy as np
import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import time

# For ordinal encoding categorical variables, splitting data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from gplearn.genetic import SymbolicRegressor

# For training random forest model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# # Step 2: Load the data
# 
# Next, we'll load the training and test data.  
# 
# We set `index_col=0` in the code cell below to use the `id` column to index the DataFrame.  (*If you're not sure how this works, try temporarily removing `index_col=0` and see how it changes the result.*)

# In[18]:


# Load the training data
train = pd.read_csv("../input/30-days-of-ml/train.csv", index_col=0, na_values=["", " "])
test = pd.read_csv("../input/30-days-of-ml/test.csv", index_col=0, na_values=["", " "])

# Preview the data
train.head()


# # Step 3: Split data and OneHot-Encode
# 
# Next, we'll need to handle the categorical columns (`cat0`, `cat1`, ... `cat9`).  

# In[19]:


train_features = train.copy()

# Preview features
print("X.shape: ", train_features.shape)
print("Original train.shape: ", train.shape)

scan_cols = train.drop(columns=['target']).select_dtypes([float, int])

# Remove outliers
for i,col in enumerate(scan_cols):
    index = train_features[col][(train_features[col] - train_features[col].mean()).abs() > 3*train_features[col].std()]
    print("%s scanned:" % col, index.shape)
    train_features.drop(index=index.index, inplace=True)

print(train_features.shape)

# Separate target from features
y = train_features.pop('target')


# In[25]:


# Apply one-hot encoder to each column with categorical data

object_cols = [col for col in train_features.columns if 'cat' in col]
num_cols = [col for col in train_features.columns if 'cat' not in col]

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(train_features[object_cols]))
OH_test_cols = pd.DataFrame(OH_encoder.transform(test[object_cols]))

# One-hot encoding removed index; put it back
OH_cols.index = train_features.index
OH_test_cols.index = test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X = train_features.drop(object_cols, axis=1)
num_X_test = test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X = pd.concat([num_X, OH_cols], axis=1)
OH_test = pd.concat([num_X_test, OH_test_cols], axis=1)

OH_X_train, OH_X_valid, OH_y_train, OH_y_valid = train_test_split(OH_X, y, train_size=0.9, test_size=0.1, random_state=0)

print(OH_X.head())

print(OH_test)


# # Step 4: Train a model
# 
# Now that the data is prepared, the next step is to train a model.  

# In[ ]:


est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(OH_X_train, OH_y_train)


# In[22]:


print(est_gp._program)


# In[24]:


predictions = est_gp.predict(OH_X_valid)
mean_squared_error(OH_y_valid, predictions)
