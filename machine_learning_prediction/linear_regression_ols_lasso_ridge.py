# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import matplotlib.pyplot as plt
import seaborn as sns
from study_analysis import normality_tests
from sklearn.metrics import r2_score

pd.set_option("display.max_columns", 20)


## Assumptions
# 
# - The residuals must be normally distributed
# - TODO: More?


def load_df(error=False):

    df = pd.read_csv("~/Code/data/SpotifyDataset160k.csv", na_values=[" ", ""])
    
    ## If you want to use standard scaling this condition must be met.
    ## Zero has a special meaning and would skew results
    if error and df.isin([0]).any().any():
        raise ValueError("Dataframe contains Zeros! Please be vary when using StandardScaler!")

    df = df.drop(df[df.danceability == 0].index)
    df = df.drop(df[df.energy== 0].index)
    df = df.drop(df[df.instrumentalness == 0].index)
    df = df.drop(df[df.liveness == 0].index)
    df = df.drop(df[df.speechiness == 0].index)
    df = df.drop(df[df.tempo == 0].index)
    df = df.drop(df[df.valence == 0].index)
    
    # we drop out some vars to reduce complexitiy
    df = df.drop(["acousticness", "year", "artists", "key", "name", "id", "release_date", "mode", "loudness"], axis="columns")
    X = df.copy().drop("popularity", axis="columns")
    y = df["popularity"]
    
    return df, X, y

def transform_df(df, X, y, scaler="minmax"):
    # TODO 2: K-Folds machen
    scan_cols = X.select_dtypes([float, int])
    scan_cols = scan_cols.drop("explicit", axis="columns")
    
    from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, MinMaxScaler
    if(scaler == "robust"): s = RobustScaler() # Use if data is already approx Normal-distributed and if 0 has no meaning and if outliers are present
    elif(scaler == "standard"): s = StandardScaler() # Use if data is already approx Normal-distributed and if 0 has no meaning and if no big outliers are present
    elif(scaler == "log"): 
        # - LogTransform() # Use this if the data is highly skewed or very condensed in some parts of it's range. LogTransform will stretch condensed parts and 
        #   condense stretched parts so the model can work better with the data
        ## # If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log        
        s = FunctionTransformer(np.log1p, validate=True)
    else: s = MinMaxScaler()
    
    
    # You can also use QuantileTransformers as a test ... but output is hard to interpret: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
    
    X[scan_cols.columns.tolist()] = s.fit_transform(scan_cols)
    
    return df, X ,y, s


def lm_df(df, X, y, s):
    ## Statsmodels OLS with formula
    import statsmodels.formula.api as smf
    model = smf.ols(formula="popularity ~ danceability + duration_ms + energy + explicit + instrumentalness + liveness + speechiness +  tempo + valence", data=X.join(y)).fit()
    
    
    
    ## Validate assumptions: 1.Normality of residuals 
    df_coefficients = pd.DataFrame(model.params).T
    df_coefficients = df_coefficients.rename(index={0: "OLS"})
    df_coefficients["rsquared_adj"] = model.rsquared_adj
    
    #normality_tests.run_normality_test(model.resid)
    #model.summary()
    
    ## Lasso tries to penalize covariates as much as possible
    ## Since it uses L1 regularization ist MUST be standardized
    from sklearn.linear_model import Lasso
    clf = Lasso(alpha=1).fit(X,y) # alpha is the penalty. 0 is OLS. 1 pure Lasso
    df_coefficients = df_coefficients.append(pd.Series([clf.intercept_, *clf.coef_], index=model.params.index, name="Lasso"))
    df_coefficients.loc["Lasso", "rsquared_adj"] = r2_score(y, clf.predict(X))
    
    
    # Ridge is a trade-off. It does not force coefficients to zero, but just minimizes them.
    ## Since it uses L1 / L2 regularization ist MUST be standardized
    from sklearn.linear_model import Ridge
    clf = Ridge(alpha=1).fit(X,y) # alpha is the penalty. 0 is OLS. 1 pure Ridge
    df_coefficients = df_coefficients.append(pd.Series([clf.intercept_, *clf.coef_], index=model.params.index, name="Ridge"))
    df_coefficients.loc["Ridge", "rsquared_adj"] = r2_score(y, clf.predict(X))

    print("Coefficients: ", df_coefficients)
    predictions = clf.predict(X)
    print("Ridge Predictions: ", predictions)
    return df,X,y,s

df, _, _ = load_df()
lm_df(*load_df())

lm_df(*transform_df(*load_df(), scaler="minmax"))
lm_df(*transform_df(*load_df(), scaler="standard"))
lm_df(*transform_df(*load_df(), scaler="log"))


## Conclusion: What we can see here is that the transformation HIGHLY alters 
## the coefficients in a non-linear and loosing their ratio to one another.
##
## Therefore transformations are NOT useful if you want to interpret the coefficients
## in a causal way.
#
## However some transformations may be helpful. For instance a log if you have 
## an exponential value like income. To achieve some sort of linear behaviour 
## you might transform this single variable. Another option is to use interactions
## which might model a multiplicative effect with a better reasoning.
## This is really highly problematic and should always be decided on a case-by-case!
## If NONE of your variables can be sensibly log-transformed, it might 
## be better to pursue some non-linear regression such as splines.



## In machine learning it is very common to transform all covariates but not 
## the target variable.
## In explanatory analysis we often only transform single variables ... but
## here also sometimes the target variable.