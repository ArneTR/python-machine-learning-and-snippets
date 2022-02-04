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



## Step #1: Data Loading and cleaning
# ---------------------------------------------------------------------------------

import pandas as pd; pd.set_option('mode.chained_assignment','raise');  pd.set_option("display.max_columns", 20);
df = pd.read_csv("~/Code/data/TitanicDataset_Train.csv", na_values=["", " "])

df.isnull().sum()

X = df.copy()

# Data Cleaning
X.pop("PassengerId")
X.pop("Cabin")
X.pop("Name")
X.pop("Ticket")

# Fill NAs on age with the column mean and remove NAs from Embarked
X.Age = X.Age.fillna(X.Age.mean()) # Alternative: X = X.dropna(subset=["Age"], axis="index")
X = X[X.Embarked.notna()] # Alternative: X = X.dropna(subset=["Embarked"], axis="index")

X.Sex = X.Sex.map({'male':0, 'female':1}) # Alternative: OrdinalEncoder, which can map back 
X.Embarked = X.Embarked.factorize()[0] # If too many values and we dont care, easy factorize is an option

y = X.pop('Survived')

## Step #2: Assumptions
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
from patsy import dmatrices

rc_Pclass = df.Pclass.value_counts().index[0]
rc_Sex = df.Sex.value_counts().index[0]
rc_Embarked = df.Embarked.value_counts().index[0]
rc_SibSp = df.SibSp.value_counts().index[0]
rc_Parch = df.Parch.value_counts().index[0]
rc_Ticket = df.Ticket.value_counts().index[0]
rc_Cabin = df.Cabin.value_counts().index[0]
rc_Embarked = df.Embarked.value_counts().index[0]

# Ticket and Cabin have just too many categories. They are not included here or later

formula = ("Survived~"
           "C(Pclass, Treatment(reference=%d))"
           "+C(Sex, Treatment(reference='%s'))"
           "+Age"
           "+C(SibSp, Treatment(reference=%d))"
           "+C(Parch, Treatment(reference=%d))"
           "+Fare"
           "+C(Embarked, Treatment(reference='%s'))"
            
            ) % (rc_Pclass, rc_Sex, rc_SibSp, rc_Parch, rc_Embarked)

formula
y, X = dmatrices(formula, data=df, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif

## In the example, all looks good!
## Note for future: It is sometimes very tricky to assess the VIF if just ONE of the 
## possible values in a category exceeds the threshold. Here gVIF (generalizedVIF) 
## should be used. This can only be done in R at the moment


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
        print(df[c].value_counts().tail(5))

if(X.shape[1]*10 > X.shape[0]*10):
    print("Very low observation count: %d. Logit model might be unusable (unstable / not work at all)" % X.shape[0])


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

# We can see, that we have outliers in age and fare on the top end. However, they are
# quite plenty and also can not be attributed to some measurement-error or any problem with
# the data.

# The next step would be to check if the outliers are isolated
X.Age.hist()
X.Fare.hist()

# We can see that the age outliers are not isolated. They are just the top end
# of a natural progression of the variable.

# Fare is a bit more complicated. Here we have at least the 400+ prices quite
# isolated from the rest.
# If we include Fare in the covariates just like that we might run into issues.
# From hereon there are some typical ways to handle this:
    # Drop the rows (would be acceptable, since it is very few rows. But makes the model less general)
    # Cap the rows (we could cap all prices at 200+ . This is a very typical design decision, as it allows to leave all the rows in the data and make the assumption "Fare makes no difference if it is 200 or 500")
        # When doing such a decision it is best validated with a second model, that only includes the 200+ fares and see of the coefficients are different there
    # Do not alter the rows at this moment, but rather check if they come up as multi-dimensional outliers

# We will first go into the multi-dimensional analysis, before we make a conclusion on Fare
    


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
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + C(Embarked) + Fare'
y_outliers, X_outliers = dmatrices(formula, data=df, return_type='dataframe')
glm_res = GLM(y_outliers, X_outliers, family=families.Binomial()).fit() # it is very important to use the formula here and not using dataframe directly, cause we need to dummy-code the categories
cooks = glm_res.get_influence().cooks_distance

cook_threshold = 4 / len(X_outliers)
print(f"Threshold for Cook Distance = {cook_threshold}")

X_outliers['cooks_d'] = cooks[0]
print( (cooks_outliers := X_outliers[X_outliers.cooks_d > cook_threshold]) ) # Inspect if we spot something odd

prop_extreme = round(100*(len(cooks_outliers) / len(X_outliers)),1)
print(f'Proportion of highly influential outliers = {prop_extreme}%')


## Summary for Outlier Assumption:
# - Age is ok and does not need to be handled
# - Multi-Variate Outliers detected not too many, so there is no further action required
# - Fare could be capped. We will try both models later


## Addon: 
## Just for the curious ones: If we use an IsolationForest we get VERY different results
## from Cook's distance. Here is the code:
from sklearn.ensemble import IsolationForest
import numpy as np
model = IsolationForest(n_estimators=100, contamination=.025)
predictions = model.fit_predict(X.drop(['SibSp', 'Parch'], axis=1).values)
outlier_index = np.where(predictions==-1) # Beware, this functions returns positions, not index values!
values = df.iloc[outlier_index[0]]
print(values)


from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from patsy import dmatrices
from scipy import stats

y_dm, X_dm = dmatrices(formula, data=df, return_type='dataframe')
logit_model = GLM(y_dm, X_dm, family=families.Binomial())
logit_results = logit_model.fit()
print(logit_results.summary())


## A good way to interpret cooks distance is, if the total amount of highly influential 
## data is too high, we might not be able to get a stable / discriminative model
## However, with low values we just continue with the model


## Assumption #5: Linearity of independent variables and log odds



## Step #3: Further data cleaning based on assumptions results
#--------------------------------------------------------------------------
# Assumption #1 Showed, that Cabin and Ticket cannot be checked. Thus we must omit them
# Assumption #2 is satisfied. Data is independent
# Assumption #3: We can see, that the columns PassengerID, Name, SibSp and Parch can not be used.
# PassengerID was useless anyway :). Name could be transformed to only last-name however or a 
# flag could be set that says: "Other family member on board"

# Resulting set: Pclass, Sex, Age, Embarked, Fare










## Step #4: Modelling and calculation


## SKLearn - Useful for getting score etc.
# The result is really horrible though
from sklearn import linear_model as lm

# if you want to add different terms like interactions, you have to
# include the following:
from patsy import dmatrices
y, features = dmatrices('Survived ~ C(Pclass) + C(Sex) + Age + C(Embarked) + Fare', X.join(y))

# otherwise just simply use the whole feature matrix:
# y = X.pop("Survived")

model = lm.LogisticRegression()
results = model.fit(features,y)
print(model.intercept_)
print(model.coef_)

model.score(features,y) # This is the accuracy, not McFadden Pseudo R2

## statsmodels - Way better if you want to see statistical data
import statsmodels.formula.api as smf
model = smf.logit(formula='Survived ~ C(Pclass) + C(Sex) + Age + C(Embarked) + Fare', data=X.join(y), missing='raise').fit() # Categorical / factorized Variables must be marked, because they are sometims NOT correctly detected
# if writing Age:C(Sex) it would ONLY use the interaction
print(model.summary()) # McFadden Pseudo R2 ... otherwise it would be model.rsquared

# The coefficients we see here are the Log-Odds. They are not probabiliets or absolute values! More on that later ...

# If doing OLS, one would just use R2 to access the accuracy, or do model.predict(X) and the do a mean_squared_error on the results
# For binary results, we just calculate the "hits" ratio

predictions = model.predict(X)
predictions = [1 if i > 0.5 else 0 for i in predictions]
print("Accuracy: ", 1-(y != predictions).mean())

## Calculate the odds-ratio / aka back-transforming of covariates
import numpy as np
conf = model.conf_int()
conf.columns = ['5%', '95%']
conf['Odds'] = model.params
#conf['Odds-Ratio-5%'] = model.conf_int()[0]
#conf['Odds-Ratio-95%'] = model.conf_int()[1]
#conf[['Odds-Ratio', 'Odds-Ratio-5%', 'Odds-Ratio-95%']] = np.exp(conf[['Odds-Ratio', 'Odds-Ratio-5%', 'Odds-Ratio-95%']])
print(model.summary())
print(np.exp(conf))
# Note in this example we have multi-categories Embarked(S,C,Q) = (0,1,2). The odds ratio here ONLY tells us the
# odds for 1 or 2 over 0. But NOT from 1 over 2. This is typical for a regression result, as 0 is the base group here.
# (The values 0,1,2 are because of the factorization and can be custom modified)

# Looking at Embarked = 1 we would say: Having the class Embarked = 1 has a 1.73-times increased odds of Surviving compared to 
# Embarked = 0
# It does NOT mean, that the probability is 1.73-times as high!!!!

# For continuous variables the interpretation is a bit tricky, as it leads to exponential effects, when we look
# at the odds instead of the log-odds
# Looking at Age we would say: For every year your odds drop by a factor of 0.96 to Survive. 
# Another way to put it: With every year the odds that you survice are 0.96-times more (or better less :) ) likely
# Note: It is not the probability that is 0.96-times less. But the odds!

# It may be better to underline this with a prediction on a reduced model

X_reduced = X.loc[:,['Age', 'Survived']]
reduced_model = smf.logit(formula='Survived ~ Age', data=X_reduced, missing='raise').fit() # Categorical / factorized Variables must be marked, because they are sometims NOT correctly detected    
X_reduced['prediction'] = reduced_model.predict(X_reduced)
X_reduced.query('prediction > 0.5') # no result ?! bad model ...

print(odds_ratio := np.exp(reduced_model.params))
# We see that the odds-ratio is 0.988037 ... so it gets less likely to survive with age.
# But so a one unit increase in age leads to a 0.988037 reduced odds ratio. 
# IMPORATNT: But a 10 unit increase is not 10*0.988037 but it is rather 0.988037**10

X_reduced.sort_values('Age').iloc[40:400:2]
# We see that for an age of 6, your probabilites to survive are 0.450802 and for age 7 it is 0.447824
# And for age 16 it is 0.421217
# Note that we cannot reach this value by doing 0.450802 * 0.988037**10 !
# We must first transform to odds, then multiply there, then backtransform:
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


## Step #5: Assession goodness of fit / Model performance


import matplotlib.pyplot as plt
## Plotting multiple plots same figure
fig, (axL, axR) = plt.subplots(2, figsize=(15, 15))
plt.suptitle("Logistic Regression Residual Plots \n using Seaborn Lowess line (N = 400)")


# Deviance Residuals
sns.regplot(model.fittedvalues, model.resid_generalized, ax= axL,
            color="black", scatter_kws={"s": 5},
            line_kws={"color":"b", "alpha":1, "lw":2}, lowess=True)

axL.set_title("Deviance Residuals \n against Fitted Values")
axL.set_xlabel("Linear Predictor Values")
axL.set_ylabel("Deviance Residuals")

# Studentized Pearson Residuals
sns.regplot(model.fittedvalues, model.resid_pearson, ax= axR,
            color="black", scatter_kws={"s": 5},
            line_kws={"color":"g", "alpha":1, "lw":2}, lowess=True)

axR.set_title("Studentized Pearson Residuals \n against Fitted Values")
axR.set_xlabel("Linear Predictor Values")
axR.set_ylabel("Studentized Pearson Residuals")

plt.show()


## Step #6: Optimize Model and iterate
# -----------------------------------------------------------

# Next logical steps would entail to think if interactions are possible and model these
# For instance cluster ages into groups? The look at sex in these age groups? Still the 
# same trend in all age groups?
# Also maybe non linear factors are involved?
# Think about the research question: What do we want to falsify?
# And NOT what does the data rectify to suggest.


# Cap Fare to 200+ or 300+


## Compare with XGBOOst and its ROC / AUC

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X,y).score(X,y)