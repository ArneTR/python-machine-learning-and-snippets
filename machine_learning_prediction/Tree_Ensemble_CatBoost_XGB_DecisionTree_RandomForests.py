# Familiar imports
import pandas as pd; pd.set_option('mode.chained_assignment','raise');

import numpy as np
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import optuna



## Important: No nan-removal, deduplication, outlier-removal, 
## feature-engineering etc. is 
## done here for simplicity, but is absolutely crucial!

# Be careful with outliers.
# Removal MIGHT help if you suspect them to be irregular.
# But if outliers are highly discriminative they are very helpful in
# Tree based models, cause they can be easily separated


df = pd.read_csv("~/Code/data/diabetes.csv", na_values=["", " "])

df.head(10)
df.describe()
df.isna().sum() # Data contains no NA values

X = df.copy() # make working copy
y = X.pop("Outcome")

# Train_Test_split is inferior to cross_val_score and is old style!
# X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.2, random_state = 0)

## Also cross_val_score and multiple folds get complicated quickly if you need
## to transform data. Therefore we use pipelines.


def objective_RandomForest(trial):
    model = RandomForestClassifier(
            n_estimators = trial.suggest_int("n_estimators", 100, 12000, step=100),
            max_depth = trial.suggest_int("max_depth", 1, 5),    
            min_samples_split = trial.suggest_int("min_samples_split", 2, 5),    
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5),    
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 1, 100),
    )
    return objective(trial, model)

def objective_DecisionTree(trial): 
    
    model = DecisionTreeClassifier(
        max_depth = trial.suggest_int("max_depth", 1, 5),    
        min_samples_split = trial.suggest_int("min_samples_split", 2, 5),    
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5),    
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 1, 100),
    )        
    return objective(trial, model)

def objective_CatBoost(trial):
    model = CatBoostClassifier(
        depth = trial.suggest_int("depth", 3,50),
        iterations = trial.suggest_int("iterations", 100, 2000, step=100),
        task_type="GPU", 
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)    
    )
    return objective(trial, model)

def objective_XGB(trial):
    model = XGBClassifier(
        tree_method='gpu_hist', # FOR GPU - Omit for CPU
        predictor='gpu_predictor', # FOR GPU - Omit for CPU
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators = trial.suggest_int("n_estimators", 100, 12000, step=100),
        max_depth = trial.suggest_int("max_depth", 1, 5),
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        gamma = trial.suggest_float("gamma", 0.1, 1.0, step=0.1),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 7),
        subsample = trial.suggest_float("subsample", 0.2, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0),
        reg_alpha = trial.suggest_float("reg_alpha", 1e-6, 100.),
        reg_lambda = trial.suggest_float("reg_lambda", 1e-6, 100.),
    )
    return objective(trial, model)

def objective(trial, model):
    

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())
        ]
    )
    
    
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, X.drop("Pregnancies", axis="columns").columns),
            ("cat", categorical_transformer, ["Pregnancies"]), # This is NOT a real categorical column, but an ordinal one. But we treat it as categorical for examplary purpose
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        #('transform', StandardScaler()), # if you want to apply to all colums, this would be a valid call
        ## Remember: Pipelines ALWAYS pass y through unchanged and transforms all X (if not restricted through ColumnTransformner): 
        ## https://stackoverflow.com/questions/18602489/using-a-transformer-estimator-to-transform-the-target-labels-in-sklearn-pipeli

        ('model', model)
    ])
    
    pipeline.fit(X,y)

    # An example with a supplied CV
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

    # if no params are supplied cross_val_score uses classic 5-folds and. equal to cv=5
    # standard scoring (accuracy for classification and rmse for regression)
    scores = cross_val_score(pipeline, X, y, cv=5, error_score='raise')
    
    ## This is the equivalent of calling:
        #s = StandardScaler() 
        #X_trans = s.fit_transform(X_train)
        #models[i].fit(X_trans, y_train)
        #models[i].score(s.transform(X_test), y_test)
    ## Notice how for scoring fit_transformed is NOT called again. Should it be? No
    ## This is in line with the explanaition here: https://scikit-learn.org/stable/modules/cross_validation.html
    ## "Data transformation with held out data"

    return np.mean(scores)

    
#['Score for RandomForest: 0.770 (0.028)', 'Score for DecisionTree: 0.715 (0.040)', 'Score for CatBoost: 0.771 (0.040)', 'Score for XGB: 0.751 (0.019)']


# If you do NOT want to do an optuna study, just use the code from inside objective()
# Just the model params must be replaced with fixed values and you can choose the model you want or 
# loop over all models
study = optuna.create_study(direction="maximize") # use direction="minimize" for Regressors / RMSE
study.optimize(objective_XGB, n_trials=100)

params = study.best_params
best_score = study.best_value
print(f"Best score: {best_score}\n")
print(f"Optimized parameters: {params}\n")

