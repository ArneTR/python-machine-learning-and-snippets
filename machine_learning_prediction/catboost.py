# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

data = pd.read_csv("~/Code/data/diabetes.csv", na_values=["", " "])

data.head(10)

features = data.copy() # make working copy

features.isna().sum() # Data contains no NA values

y = features.pop("Outcome")


X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.2, random_state = 0)



# now we compare the performances for XGBoost and CatBoost


print("XGBClassifier Train Performance: ", XGBClassifier(use_label_encoder=False).fit(X_train, y_train).score(X_train, y_train))
print("XGBClassifier Test Performance: ", XGBClassifier(use_label_encoder=False).fit(X_train, y_train).score(X_test, y_test))
print("XGBClassifier Test Performance: ", XGBClassifier(n_estimators = 10, max_depth=5, colsample_bytree = 0.3, alpha= 10, learning_rate=0.1, objective="reg:linear", use_label_encoder=False).fit(X_train, y_train).score(X_test, y_test))

## Important
# more details on last model 
# These data ONLY makes sense when using Regressor. NOT Classifier!!!!
#
model = XGBClassifier(n_estimators = 100, max_depth=3, use_label_encoder=False, eval_metric="logloss").fit(X_train, y_train)
y_pred=model.predict(X_test)
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))

print("Catboost TrainingPerformance: ", CatBoostClassifier().fit(X_train, y_train).score(X_train, y_train))
print("Catboost Test Performance: ", CatBoostClassifier().fit(X_train, y_train).score(X_test, y_test))
