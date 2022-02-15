#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 23:00:29 2022

@author: light
"""

files = ["TitanicDataset_Train.csv",
"claudia_schlaf.csv",
"nlsw88.csv",
"CitesforSara.csv",
"accidents.csv",
"customer.csv",
"steph_curry_shot_data.csv",
"SpotifyDataset160k.csv",
"autos.csv",
"california_housing.csv",
"House_Prices_and_Crime.csv",
"Gender_StatsData.csv",
"power_plant.csv",
"kevin_durant_shot_data.csv",
"NetflixOriginals.csv",
"fastfood.csv",
"concrete.csv",
"Pokemon.csv",
"adverse_food_events_kaggle.csv",
"state_wages_fake.csv",
"arne_schlaf.csv",
"example_wp_log_peyton_manning.csv",
"census80.csv",
"hunger_data.csv",
"qian.csv",
"indiv_final.csv",
"diabetes.csv",
"gb_food.csv",
"lebron_james_shot_data.csv",
"arne_schlaf_copy.csv",
"30-days-of-ml_train.csv",
"Bihar_sample_data.csv",
"30-days-of-ml_test.csv",
"final_exam_data_neymans_inference.csv",
"demo_dataset_constant_variable.csv",
"sample_data_1.csv",
"30_days_of_ML_Train.csv",
"Social_Network_Ads.csv",
"teachers_final.csv",
"histogram1.csv"]


import pandas as pd
file = files[2]
print(file)
df = pd.read_csv("/Users/light/Code/data/"+file)
print(df.head())
df.black.value_counts()


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X = df.drop('black', axis='columns')
y = df.black


X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3)

xgb = XGBClassifier( use_label_encoder=False)
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)


