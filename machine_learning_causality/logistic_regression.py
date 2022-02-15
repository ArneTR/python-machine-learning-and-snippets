## What is logistic regression?
# Simply put: instead of constructing and equation like y ~ b0 + B*X 
# we use an equation that follows the logistic function
# which is: y = e^(b0 + B*X) / (1 + e^(b + B*X))
# Note: B and X are vectors

# So we are not saying: "We suspect a linear relation, so we fit the parameters
# in a way, that the equations works."
# But rather: "WE are assuming a logisitc relation. Therefore we plug the parameters
# into this function and make it fit"

# To make the fitting easier, we ln() on both sides. Resulting in:
# ln(y)/ln(1-y) = b0 + B*X

# The logit function is a function that transforms the categorical nature of y
# to a continuous nature on the range of 0-1
# More precisely: it is the odds ratio that the value of Y occurs given 
# all possible occurences of Y. So it is the rate (normalized to 0-1).
# Mathematically correct: It is the log(P/1-P) .. P is the probability

# Technically this is just one way of doing it. So it is just a definition, not
# a "logical" way of doing it. One can also use probit, arcsin() etc.
# But it has evolved, that logit is the easiest to interpret.

# We could the the probability instead of the odds, but this value is not constanct
# over the range of the covariate. Therefore we use the odds, cause it is a constant
# number and thus easier to understand and work with

# The interpretation is different from a linear model, where we say: being in 
# group A increases the result by 3.4: 

#   # Categorical: The odd of having the outcome "1" for group-A is ##.## times that of group-B

#   # Continuous The odds of the outcome increases/decreases by a factor of ##.## for 
#   # every one unit increase in the independent variable.

# Note on odds: Odds are a ratio of the probability. It basically is defined as:
    # O(H) = P(H) / 1-P(H)  # H being the hyptohesis. And O(H) meaning the odds of H
# You can also use probabilites in Log-Regression, but odds are constant over the
# range of the covariates values (because they are a ratio). Probabilites change over the range.    

# Note #2: Odds and Odds-Ratio are often used interchangebly in literature. The mean the same.



## Step #1: General Imports and data Loading
# ---------------------------------------------------------------------------------

import pandas as pd; pd.set_option('mode.chained_assignment','raise');  pd.set_option("display.max_columns", 20);
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import sklearn.linear_model
from patsy import dmatrices
import scipy.stats
import numpy as np


df = pd.read_csv("~/Code/data/TitanicDataset_Train.csv", na_values=["", " "])


## Step #2: Data Cleaning
# -------------------------------------------------------------------------------------

df.isnull().sum() # check for nulls

df.nunique() # cardinality check

## check for unplausible values
(df.Age <= 0).any()
(df.Fare <= 1).any() # Ticket prices below 1 USD?

X = df.copy()

# We drop the Fare=0 rows. They make no sense and are just very few
X = X.drop(X[df.Fare == 0].index)


# Data Cleaning
X.pop("PassengerId") # No info contained due to max-cardinality
X.pop("Cabin") # Very high cardinality and high null count
X.pop("Name") # Very low info contained unprocessed.  We leave it out for this example
X.pop("Ticket") # Very high cardinality. Possible option would have been to reprocess colum to binary and make 0 for unique ticket and 1 for shared ticket. We drop for now

# Fill NAs on age with the column mean and remove NAs from Embarked
X.Age.hist() # Since Age is a bit skewed, we rather go for the median, to avoid bias
X.loc[:, 'Age'] = X.Age.fillna(X.Age.median()) # Alternative: X = X.dropna(subset=["Age"], axis="index")
X = X.dropna(subset=["Embarked"], axis="index")

X.loc[:, 'Sex'] = X.Sex.map({'male':0, 'female':1}) # Alternative: OrdinalEncoder, which can map back 
X.Embarked = X.Embarked.factorize()[0] # If too many values and we dont care, easy factorize is an option


y = X.pop('Survived')



## Step #3: Decide on a model
# -------------------------------------------------------------------------------------

# We must decide on a model early on, as some of the methods coming next will 
# need a model to assess the assumptions
# 
# If we have a research question at this point we would create the model with
# this equation.
# Example: Do women have a higher survival rate when grouped by age?
#   would equate to Survived ~ Age + Sex + Age:Sex
# etc.
#
# For the example we will go with the approach, that we have NO research question 
# and we just wanna find a model that balances generality and prediciton performance.
# This means, that we do not include all possible combinations and interactions
# and maybe also second order terms, but we just include all covariates
# without any interactions in the first run

formula = 'Survived ~ C(Pclass) + C(Sex) + Age + C(Embarked) + Fare + SibSp + Parch'


## Step #4: Assumptions
#------------------------------------------------------------------------------------

# As with every model, assumptions must be checked.

# Since Logistic Regression is just a case of the "Generalized Linear Model" we do not 
# have the assumption, that the response variable follows a gaussian distribution. 
# Also the residuals do not have to (and actually cannot due to the link function) follow a normal distribution. (vgl. https://www.pythonfordatascience.org/logistic-regression-python/)
# Homoscedasticity is also not needed.
 

## Assumption #1: No Multicolineratity
##
## Here we can use variance_inflation_factor, which is already implemented in statsmodles
## Effectively we are regression one covariate against all others (target removed!)
## Important: The statsmodels implementation excepts a provided constant, which will be used as the intercept! This is needed! 
## 
## Rules: VIF > 5 (also in some sources > 10) means colinear
## 
## The MOST frequent variable should be used as the reference category!
from statsmodels.stats.outliers_influence import variance_inflation_factor

rc_Pclass = X.Pclass.value_counts().index[0]
rc_Sex = X.Sex.value_counts().index[0]
rc_Embarked = X.Embarked.value_counts().index[0]
rc_SibSp = X.SibSp.value_counts().index[0]
rc_Parch = X.Parch.value_counts().index[0]

formula_vif = ("Survived~"
           "C(Pclass, Treatment(reference=%d))"
           "+C(Sex, Treatment(reference=%d))"
           "+Age"
           "+C(SibSp, Treatment(reference=%d))"
           "+C(Parch, Treatment(reference=%d))"
           "+Fare"
           "+C(Embarked, Treatment(reference=%d))"
            
            ) % (rc_Pclass, rc_Sex, rc_SibSp, rc_Parch, rc_Embarked)

print(formula_vif) # debug

_, X_vif = dmatrices(formula_vif, data=X.join(y), return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif["features"] = X_vif.columns
vif

''' In the example, all looks good

VIF of Intercept is always ignored.

It should be noted that there is different voices online if VIF can be used
for categorical variables at all, or just in limited cases. 
https://www.researchgate.net/post/Anyone_familiar_with_VIF_Variance_Inflation_Factor_and_categorical_variables

The limited case is when you run it like here with the reference category supplied.
From my research it may not be run without and even in this cases it will get 
un-interpretable if just ONE of the possible values in a category exceeds the threshold. 
Here gVIF (generalizedVIF) should be used. 
Out of the box this can only be done in R with the car package at the moment
'''

## Assumption #2: Indepence
##
## This is usually not checked programmatically but rather through Intuition.
## At least when the data generation process is known.
## This means: No repeated measurements & no matched pairs
## Also there should be no "derived" variables that through some linear / non-linear
## combination can be created from another
##
## Important: If there is NO domain knowledge about the covariates I will have to
## dig deeper and improve this point. For now I only encountered data with known generation process.

## However, a very quick and easy analysis is to plot the residuals over time.
## Although this is only tells us IF there is indepence, and cannot conclude that they are 
## independent (they still might be dependent), it is very helpful

# Generate residual series plot

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
logit_model = GLM(y, X, family=families.Binomial())
logit_results = logit_model.fit()

ax = sns.lineplot(X.index.tolist(), scipy.stats.zscore(logit_results.resid_deviance))
ax.set_xlabel("Index")
ax.set_ylabel("Residual Deviance")


''' 
The plot should look like white noise. If it does not and you see any pattern
it is very likely not independent
'''

## Assumption #3: Sample Size
##
## Logistic regression typically requires a large sample size.  
## A general guideline is that you need at minimum of 10 cases with the least 
## frequent outcome for each independent variable in your model. 
## For example, if you have 5 independent variables and the expected probability 
## of your least frequent outcome is .10, then you would need a minimum sample 
## size of 500 (10*5 / .10).
##
## The other rule of thumb is to have at least 10 observations per covariate in
## general. As this is important for real-valued unbounded variables

for c in df.columns:
    if df[c].value_counts().tail(1).iloc[0] < 10 :
        print("Warning! %s is below threshhold of 10. Column type is %s" % (c, df[c].dtype))
        print("Unique values: ", df[c].nunique())
        print(df[c].value_counts().head(2))
        print(df[c].value_counts().tail(3))


# And we also check for a general low obeservation count
if(X.shape[1]*10 > X.shape[0]*10):
    print("Very low observation count: %d. Logit model might be unusable (unstable / not work at all)" % X.shape[0])

'''
From the code we see that SibSp and Parch can be problematic. We reduce our model in this case.
Usually furtter analysis if the column is helpful at all would be neeed and 
maybe even dropping some rows should be considered
'''

formula = 'Survived ~ C(Pclass) + C(Sex) + Age + C(Embarked) + Fare'


## Assumption #4: No strong influential outliers
##
## This is argueably the assumption that needs the most experience and intuition
## to identify and handle outliers.
## The underlying problem is, that strong outliers do alter the regression line
## significantly. Hereby can only one outlier have a very strong impact. So much,
## that only one observation with an outlier can decide if a model is a really good
## approximation to reality, or totally off.
##
## Outliers can be detected in two ways:
##    - Univariate. Meaning we only look at one covariate at a time
##    - Multivariate: We look at two or more covariates simultaniously
##
## The univariate approach is often done also in data-cleaning. Sometimes it 
## is left out however, because some models like Gradient-Boosting do profit 
## from outliers, so they are left in
## Logistic Regression is sensitive to outliers, so we run both approaches

## Univariate Outliers
##      Since we only have very low or very high cardinalty categoricals we 
##      can safely ignore them. 
##      The only relevant columns are the real-valued ones

print("Age+ outliers:\n", X.Age.loc[(X.Age > (X.Age.mean() + X.Age.std()*3))])
print("Age- outliers:\n", X.Age.loc[(X.Age < (X.Age.mean() - X.Age.std()*3))])


print("Fare+ outliers:\n", X.Fare.loc[(X.Fare > (X.Fare.mean() + X.Fare.std()*3))])
print("Fare- outliers:\n", X.Fare.loc[(X.Fare < (X.Fare.mean() - X.Fare.std()*3))])

'''
We can see, that we have outliers in age and fare on the top end. However, they are
quite plenty and also can not be attributed to some measurement-error or any problem with
the data. 
'''


# The next step would be to check if the outliers are isolated
X.Age.hist()
X.Fare.hist()

'''
We can see that the age outliers are not isolated. They are just the top end
of a natural progression of the variable.

Fare is a bit more complicated. Here we have at least the 400+ prices quite
isolated from the rest.
If we include Fare in the covariates just like that we might run into issues.
From hereon there are some typical ways to handle this:
    # Drop the rows (would be acceptable, since it is very few rows. But makes the model less general)
    # Cap the rows (we could cap all prices at 200+ . This is a very typical design decision, as it allows to leave all the rows in the data and make the assumption "Fare makes no difference if it is 200 or 500")
        # When doing such a decision it is best validated with a second model, that only includes the 200+ fares and see of the coefficients are different there
    # Do not alter the rows at this moment, but rather check if they come up as multi-dimensional outliers

We will first go into the multi-dimensional analysis, before we make a conclusion on Fare
'''    


## Multi-Variate Outliers
##
## Multi-Variate Outlier Detection is a broad topic, and when researched there is a plethora
## of methods and approaches. Not all of them are relevant for Logisit Regression though.
## 
## The underlying idea is, that an outlier is not detectable only in one dimension, but rather
## only if we look at multiple dimensions separately. For example: A baby with age 2
## is not an outlier and a person driving license. But a baby with a driving license is clearly an outlier.
## The renowned R-Package "performance" has a great function with good documentation on 
## different methods for outlier detection: https://easystats.github.io/performance/reference/check_outliers.html
## Their approach is to use multiple methods at once and then compute some compound threshold 
## when a value counts as outlier.
## Many methods out there are generally good outlier detectors. But their approach does not 
## really apply to what we wanna do here in Logistic Regression assumption validation.
## For example: IsolationForest is very good at detecting outliers. However, it does
## this though a Tree-Method that detects "unusual combinations". If these combinations
## are value-wise all very mean-like values, they do not affect the internal calculations of the 
## Logistic Regression. Therefore we might get false-positives from IsolationForest.
##
## A better, although deemed "outdated" in the "performance" package is Cook's-Distance.

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

y_outliers, X_outliers = dmatrices(formula, data=df, return_type='dataframe')
glm_res = GLM(y_outliers, X_outliers, family=families.Binomial()).fit() # it is very important to use the formula here and not using dataframe directly, cause we need to dummy-code the categories
cooks = glm_res.get_influence().cooks_distance

cook_threshold = 4 / len(X_outliers)
print(f"Threshold for Cook Distance = {cook_threshold}")

X_outliers['cooks_d'] = cooks[0]
print( (cooks_outliers := X_outliers[X_outliers.cooks_d > cook_threshold]) ) # Inspect if we spot something odd

prop_extreme = round(100*(len(cooks_outliers) / len(X_outliers)),1)
print(f'Proportion of highly influential outliers = {prop_extreme}%')


''' Summary for Outlier Assumption:
 - Age is ok and does not need to be handled
 - Multi-Variate Outliers detected not too many, so there is no further action required
 - Fare could be capped. We will try both models later
'''

'''
A good way to interpret cooks distance is, if the total amount of highly influential 
data is too high, we might not be able to get a stable / discriminative model
However, with low values we just continue with the model
'''


## Addon: 
# Just for the curious ones: If we use an IsolationForest we get VERY different results
# from Cook's distance. Here is the code:

from sklearn.ensemble import IsolationForest

model = IsolationForest(n_estimators=100, contamination=.025)
predictions = model.fit_predict(X.drop(['SibSp', 'Parch'], axis=1).values)
outlier_index = np.where(predictions==-1) # Beware, this functions returns positions, not index values!
values = df.iloc[outlier_index[0]]
print(values)





## Assumption #5: Linearity of independent variables and log odds
## 
## The agreed method to check this assumptions is the box-tidwell test
##
## This tests creates a non-linear transformation and then compares the linear model
## With the non-linear one. If the non-linear model has a better fit we have valid
## reason to believe, taht the log-odds of the target and the covariates are not linear

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families


(X == 0).any() # Box tidwell test will fail if we have 0 values in real-valued columns
''' 
All good, cause we cleaned fare at the beginning
'''

formula_bt = 'Survived ~ Age + Age:np.log(Age) + Fare + Fare:np.log(Fare)'
y_bt, X_bt = dmatrices(formula_bt, data=X.join(y), return_type='dataframe')
logit_results = GLM(y_bt, X_bt, family=families.Binomial()).fit()
print(logit_results.summary())
'''
We only look at the interaction terms and we can see, that:
    - Age is not significant with 0.123 > 0.05
    - Fare is significant with 0.000

Therefore fare should either not be included, or we include a qudaratic term,
or even higher polynomial of it
'''


## Another good way to see this is to visualize the log-odds with the covariates
y_1, X_1 = dmatrices("Survived ~ C(Pclass) + C(Sex) + Age + C(Embarked) + Fare", data=X.join(y), return_type='dataframe')
logit_results = GLM(y_1, X_1,  family=families.Binomial()).fit()
print(logit_results.summary())
predicted = logit_results.predict(X_1)
log_odds = np.log(predicted / (1 - predicted))

# Visualize predictor continuous variable vs logit values (Age)
plt.scatter(x = X_1['Age'].values, y = log_odds);
plt.show()
''' You can clearly see the linear relationship '''


# Visualize predictor continuous variable vs logit values (Age)
plt.scatter(x = X_1['Fare'].values, y = log_odds);
plt.show()
''' You can clearly see, that there is no linear relationship '''





## Addon: It should be noted that there is also a different and often cited method
## to analyse the linearity through the residuals.
## This is shown here: https://freakonometrics.hypotheses.org/8210
## Interpretation is often very difficult though, as it relies on a visual check
## that talks about a "almost horizontal line". Not very specific sadly ...



## Step #5: Further data cleaning based on assumptions results
#--------------------------------------------------------------------------
'''
- Preface: Cabin, Ticket and PassengerID could not be used due to high cardinality
- Assumption #1 of Multi-Colinearity is satisfied.
- Assumption #2 is satisfied. Data is independent
- Assumption #3 of Sample Size showed that SibSP and Parch are problematic. From domain knowledge we know that we could reprocess this data, as it relates to traveling with family. For this example we assume no domain knowledge and just remove it from the model.
- Assumption #4 of strong influential outliers showed no critical amount of outliers. This is concidered satisfied. Depending on what you want to achieve with the model (like life and death situations with privacy / medical data) and how strong your claims should be in the end, this might be debateable and further analysis would be needed.
- Assumption #5 Fare was also not linear, so it will be omitted
- Assumption #6 (not a single chapter in this notebook) is satisfied, as it states that the target variable has to be binary. We knew this in advance and did not check it. 
 
 Resulting formula: **Survived ~ C(Pclass) + C(Sex) + Age + C(Embarked)**
'''

formula = 'Survived ~ C(Pclass) + C(Sex) + Age + C(Embarked)'


## Step #6: Running model and first look at coefficients

# I usually run my models with statsmodels when doing causal understanding of the data and not just prediction.

# First thing to look is if the model did converge.

# **SKLearn** is useful for easily getting and comparing scores etc.
# The result display is really horrible though
# if you want to add different terms like interactions, you have to include the following:

y_sklearn, features = dmatrices(formula, X.join(y))

# otherwise just simply use the whole feature matrix:
# y = X.pop("Survived")

model = sklearn.linear_model.LogisticRegression(fit_intercept=False, penalty='none', max_iter=500) # set intercept to false, cause patsy already includes the intercept
results = model.fit(features, y_sklearn)
print(model.intercept_)
print(model.coef_)
model.score(features,y_sklearn) # This is the accuracy, not McFadden Pseudo R2

## statsmodels - Way better if you want to see statistical data

model = smf.logit(formula=formula, data=X.join(y), missing='raise').fit() # Categorical / factorized Variables must be marked, because they are sometims NOT correctly detected
# if writing Age:C(Sex) it would ONLY use the interaction
print(model.summary()) # McFadden Pseudo R2 ... otherwise it would be model.rsquared

# The coefficients we see here are the Log-Odds. They are not probabiliets or absolute values! More on that later ...

# If doing OLS, one would just use R2 to access the accuracy, or do model.predict(X) and the do a mean_squared_error on the results
# For binary results, we just calculate the "hits" ratio

predictions = model.predict(X)
predictions = [1 if i > 0.5 else 0 for i in predictions]
print("Accuracy: ", 1-(y != predictions).mean())

## Calculate the odds-ratio / aka back-transforming of covariates

conf = model.conf_int()
conf.columns = ['5%', '95%']
conf['Odds'] = model.params
#conf['Odds-Ratio-5%'] = model.conf_int()[0]
#conf['Odds-Ratio-95%'] = model.conf_int()[1]
#conf[['Odds-Ratio', 'Odds-Ratio-5%', 'Odds-Ratio-95%']] = np.exp(conf[['Odds-Ratio', 'Odds-Ratio-5%', 'Odds-Ratio-95%']])
print(model.summary())
print(np.exp(conf))

'''
Note in this example we have multi-categories Embarked(S,C,Q) = (0,1,2). The odds ratio here ONLY tells us the
odds for 1 or 2 over 0. But NOT from 1 over 2. This is typical for a regression result, as 0 is the base group here.
(The values 0,1,2 are because of the factorization and can be custom modified)

Looking at Embarked = 1 we would say: Having the class Embarked = 1 has a 1.73-times increased odds of Surviving compared to 
Embarked = 0
It does NOT mean, that the probability is 1.73-times as high!!!!

For continuous variables the interpretation is a bit tricky, as it leads to exponential effects, when we look
at the odds instead of the log-odds
Looking at Age we would say: For every year your odds drop by a factor of 0.96 to Survive. 
Another way to put it: With every year the odds that you survice are 0.96-times more (or better less :) ) likely

Note: It is not the probability that is 0.96-times less. But the odds!
'''

# It may be better to underline this with a prediction on a reduced model
df_reduced = X.loc[:, ['Age']].join(y)
reduced_model = smf.logit(formula='Survived ~ Age', data=df_reduced, missing='raise').fit() # Categorical / factorized Variables must be marked, because they are sometims NOT correctly detected    
df_reduced['prediction'] = reduced_model.predict(df_reduced)
df_reduced.query('prediction > 0.5') # no result ?! bad model ...

print(odds_ratio := np.exp(reduced_model.params))
'''
We see that the odds-ratio is 0.988037 ... so it gets less likely to survive with age.
But so a one unit increase in age leads to a 0.988037 reduced odds ratio. 

IMPORATNT: But a 10 unit increase is not 10*0.988037 but it is rather 0.988037**10
'''

df_reduced.sort_values('Age').iloc[40:400:2]

''' 
We see that for an age of 6, your probabilites to survive are 0.450802 and for age 7 it is 0.447824
And for age 16 it is 0.421217
Note that we cannot reach this value by doing 0.450802 * 0.988037**10 !
We must first transform to odds, then multiply there, then backtransform:
'''

odds_age_6 = 0.450802 / (1-0.450802) 
odds_age_16 = odds_age_6 * odds_ratio.Age**10
probability_age_16 = odds_age_16 / (1+odds_age_16) # Which is equal to the probability we got in the dataframe earlier

# However, if we would have used the log-odds we could just add them up! But log-odds have a very quirky way of wrapping your head around them and should not be used.

# Formula for Odds => Prob: P(H) = O(H) / 1+O(H)
# Formula for Prob => Odds: O(H) = P(H) / [1-P(H)]

# Using odds as a fixed value is VERY untypical in common language. Nobody says: "The odds are 3".
# Rather it is said: The odds are 3 to 1. Thus meaning a probability of 3/4.
# However we MUST use the odds, as it is very hard to do arithmetic caclulations with probabilites in log-regression since it is not constant! Can only repeat it again!

# Further note on odds: A doubling in odds does not mean a doubling a probability. 
    # Ex: Odds are 3 (namely 6 to 2). Doubled odds are 6. 
        # Probability was initially 3/4. Now it is 6/7


## Step #7: Assession goodness of fit / Model performance
# -------------------------------------------------------------------

# If the following happens this is usually a warning sign:    
# - Getting very high standard errors for regression coefficients
# - The overall model is significant (LLR p-value), but none of the coefficients are significant

# Generally the first value to look at is the LLR p-value. This tells us if the model is 
# better than plugging in zero for all coefficients (same as F-Test in Linear Regression)
# Since our p-value is very low (9.254e-10) we can conclude that the model is better than an all-zero-model
#
# Next interesting value is the pseudo-R2
# This value is similar to the R2 from linear regression in its range (0-1).
# But pseudo-R2 rarely goes into the 0.8+ range like good linear regression models do.
# A value of 0.3286 is pretty good actually
#
# The values for Log-Likelihood and LL-Null are already expressed through pseuod-R2 (which is their quotient).
#  
# The base accuary of  0.7894736842105263 comes from a treshhold of 0.5.
# We now drill down further into advanced charateristics and finally decide on a threshhold.

model = smf.logit(formula=formula, data=X.join(y), missing='raise').fit() # Categorical / factorized Variables must be marked, because they are sometims NOT correctly detected
y_pred = model.predict(X)
y_pred_binarized = [0 if el < 0.5 else 1 for el in y_pred ]


## ROC Curvers
# ROC curves are typically used, when the target variable is balanced
len(y[y==0]) / len(y) # 0.612
# Our target variable is quite balanced, so ROC is a good choice.
# For imbalanced sets we would use Precision-Recall

import sklearn.metrics

false_positive_rate, true_positive_rate, threshholds =  sklearn.metrics.roc_curve(y, y_pred)

threshholds[0] = 1 # first threshhold value out of bounds. We set to 1 to make plot window not scale
sns.lineplot(x=false_positive_rate, y=threshholds, label="Threshhold") # ROC-Curve
x_new = [x/1000 for x in range(0,1000)]
sns.lineplot(x=x_new, y=x_new, label="Random-Guessing")
ax = sns.lineplot(x=false_positive_rate, y=true_positive_rate, label="Model") # ROC-Curve
ax.set(xlabel="1-Specificity", ylabel="Sensitivity", title='ROC-Curve')
print("AUC:", sklearn.metrics.roc_auc_score(y, y_pred)) # AUC score


test_df = X.join(y)
test_df['y_pred'] = np.asarray(y_pred_binarized)

test_df['y_diff'] = test_df.Survived - test_df.y_pred
 
test_df_y1 = test_df[test_df.Survived == 1]
test_df_y0 = test_df[test_df.Survived == 0]

sensitivity = len(test_df_y1[test_df_y1.y_diff == 0]) / len(test_df_y1)
specificity = len(test_df_y0[test_df_y0.y_diff == 0]) / len(test_df_y0)

'''
Looking at the whole model we see, that the sensitivity is at 0.7 while having
a spcificity of 0.844.
None of these values are very good. 
In medical conditions it is usually expected to have a sensitivity of > 0.99
and a specificity of > 0.9

The ROC curve also tell us, that if we wanna have sensitivty of say 0.9, this
would come with a cost of a specificity of 0.2 or below.
The threshold for this value would be around 0.1 as seen in the blue curve
'''


## Precision Recall
# Used for imbalanced target variables
# precision recall tells the ratio of all true hits over all (true hits and also true misses) 
# compared to all true hits over all true guesses (positive predictiing power)
# 

# Sensitivity: How many correct positive values predicted over all positive values in dataset
# Specificity: How many correct negatives where predcted over all negative values in the dataset

# We can compare ROC sensitvity to specificity and Precision Recall side by side
# The first one tells us how accurate our guesses are given the data. However, it does not necesssarily tell us
# how often we tried. Because if we say 100% positive, we have a sensitivity of 100%.
# But Specificity will be zero in this case.
# By using recall, we get also a very high amout, as formula is Correct Positives / (Correct Positives + Wrong Positives)

# But how does this help?
# Since it does not take into account the wrong negatives it only helps in assessing the model
# if we do not care about one of the values (0 in this setup).
# We only want to know how good we can predict class 1.
#
# The model is considered good if it bows to the top right. Also the baseline is not
# a diagonal, but a horizontal line ath the y-height of the ratio of len(y[y==0]) / len(y) # 0.612 in our case
#
# this means that the model performs best if we tune it no further than having a Recall of 0.85
sklearn.metrics.f1_score(y, y_pred_binarized)


precision, recall, threshholds_pr = pr_data = sklearn.metrics.precision_recall_curve(y, y_pred)
sns.lineplot(x=recall, y=precision)

z = threshholds_pr.tolist()
z.append(1)
sns.lineplot(x=recall, y=z) ## Recall == Sensitivty


print(sklearn.metrics.f1_score(y, y_pred_binarized))


''' Key takeaway: If the dataset is unbalanced also look at the 
Precision-Recall curve. It will give better insights.
However, if data is not unbalanced and/or you are interested in both preditctions,
always look at the ROC curve
'''







## AIC


## Probit vs. Logit

## BIC
model = LogisticRegression(solver='lbfgs', class_weight = {0:0.61, 1:1.0}, max_iter=300)
result = model.fit(X,y)
help(result.score)




# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
# define dataset
A, b = make_classification(n_samples=10000, n_features=10, n_redundant=1, n_informative=5,
	n_clusters_per_class=4, weights=[0.99], flip_y=0)
# summarize class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(b == label)[0]
	pyplot.scatter(A[row_ix, 0], A[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()


from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', weights = {0:1.0, 1:1.0})
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, A, b, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))

model = LogisticRegression(solver='lbfgs')
model.fit(A,b)
b_pred = model.predict_proba(A)[:,1]
print(sklearn.metrics.roc_auc_score(b, b_pred)) # AUC score

weights = {0:1.0, 1:2.0}
model = LogisticRegression(solver='lbfgs', class_weight=weights)
model.fit(A,b)
b_pred = model.predict_proba(A)[:,1]
print(sklearn.metrics.roc_auc_score(b, b_pred)) # AUC score


weights = {0:1.0, 1:100.0}
model = LogisticRegression(solver='lbfgs', class_weight=weights)
model.fit(A,b)
b_pred = model.predict_proba(A)[:,1]
print(sklearn.metrics.roc_auc_score(b, b_pred)) # AUC score

weights = {0:1.0, 1:101.0}
model = LogisticRegression(solver='lbfgs', class_weight=weights)
model.fit(A,b)
b_pred = model.predict_proba(A)[:,1]
print(sklearn.metrics.roc_auc_score(b, b_pred)) # AUC score

precision, recall, threshholds_pr = pr_data = sklearn.metrics.precision_recall_curve(b, b_pred)
sns.lineplot(x=recall, y=precision)

false_positive_rate, true_positive_rate, threshholds =  sklearn.metrics.roc_curve(b, b_pred)
sns.lineplot(x=false_positive_rate, y=true_positive_rate) # ROC-Curve





## Step #8: Optimize Model and iterate
# -----------------------------------------------------------

# Next logical steps would entail to think if interactions are possible and model these
# For instance cluster ages into groups? The look at sex in these age groups? Still the 
# same trend in all age groups?
# Also maybe non linear factors are involved?
# Think about the research question: What do we want to falsify?
# And NOT what does the data rectify to suggest.


# Also we should try out different interactions. Be aware that this generally 
# considered a bad practice when answering reasearch questions as it may 
# be considered as p-hacking.
# Statistically, the more "model equations" / combinations / groupings you 
# evaluate the higher possibility is, that you find some statistical significance
# just by chance.
# Therefore: The more analysis you run the more you need to make sure that what you 
# did find generalizes well.
# This can be done through:
#   - Running many K-Folds run on the data and assessing how often your claim holds
#   - Redoing the whole study and generating more data and re-check your claims on fresh data (reproducing prior findings)
#
# A general warning sign here: Large changes in coefficients when adding predictors
# This either means you have high colinearity or your model is unstable in general.

# Cap Fare to 200+ or 300+


## Compare with XGBOOst and its ROC / AUC

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X,y).score(X,y)