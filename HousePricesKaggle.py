# Familiar imports
import numpy as np
import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split


data = pd.read_csv("~/Code/data/house_prices_kaggle/train.csv", index_col="Id", na_values=["", " "])

print(data.describe())

print(data.head())

features = data.copy() # make working copy

scan_cols = features.drop(columns="SalePrice").select_dtypes([object])
for i,col in enumerate(scan_cols):
    features[col] = features[col].factorize()[0]

# Handle NA columns
for i,col in enumerate(features):
    print("Column: ", col, " has NA? ", features[col].isna().any(), ": ", features[col].isna().sum())
    features = features.dropna(subset=[col], axis="index")


# First we make a bare round of Linear Regression to get an idea if the dataset
import statsmodels.formula.api as smf
model = smf.ols(formula='SalePrice ~ MSSubClass + MSZoning + LotFrontage + LotArea + Street + Alley + LotShape + LandContour + Utilities + LotConfig + LandSlope + Neighborhood + Condition1 + Condition2 + BldgType + HouseStyle + OverallQual + OverallCond + YearBuilt + YearRemodAdd + RoofStyle + RoofMatl + Exterior1st + Exterior2nd + MasVnrType + MasVnrArea + ExterQual + ExterCond + Foundation + BsmtQual + BsmtCond + BsmtExposure + BsmtFinType1 + BsmtFinSF1 + BsmtFinType2 + BsmtFinSF2 + BsmtUnfSF + TotalBsmtSF + Heating + HeatingQC + CentralAir + Electrical  + LowQualFinSF + GrLivArea + BsmtFullBath + BsmtHalfBath + FullBath + HalfBath + BedroomAbvGr + KitchenAbvGr + KitchenQual + TotRmsAbvGrd + Functional + Fireplaces + FireplaceQu + GarageType + GarageYrBlt + GarageFinish + GarageCars + GarageArea + GarageQual + GarageCond + PavedDrive + WoodDeckSF + OpenPorchSF + EnclosedPorch  + ScreenPorch + PoolArea + PoolQC + Fence + MiscFeature + MiscVal + MoSold + YrSold + SaleType + SaleCondition', data=features, missing='raise').fit() # Categorical / factorized Variables must be marked, because they are sometims NOT correctly detected
# if writing Age:C(Sex) it would ONLY use the interaction
print(model.summary())
print(model.prsquared) # McFadden Pseudo R2 ... otherwise it would be model.rsquared


# Important features seem to be
# LotArea, Street, Utilities, Neighborhood, Condition2,


# First we should make an EDA.
# Then we should get an idea which features to use
# Then we should select features and maybe create new ones
# Then we Encode / Scale / Impute Columns or drop problematic rows

features.isna().sum() # Data contains no NA values

scan_cols = features.drop(columns="SalePrice").select_dtypes([float, int])
for i,col in enumerate(scan_cols):
    index = features[col][(features[col] - features[col].mean()).abs() > 3*features[col].std()]
    print("%s scanned:" % col, index.shape)
    features.drop(index=index.index, inplace=True)
