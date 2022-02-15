import pandas as pd

## What is logistic regression?
# Simply put: instead of y ~ a+b+c ... it transforms y so that: 
# logit(y) = a+b+c+d
#
# The logit function is a function that transforms the categorical nature of y
# to a continuous nature on the range of 0-1
# More precisely: it is the probability that the value of Y occurs given 
# all possible occurences of Y. So it is the rate (normalized to 0-1).
# Mathematically correct: It is the log(P/1-P) .. P is the probability
#
# Technically this is just one way of doing it. So it is just a definition, not
# a "logical" way of doing it. One can also use probit, arcsin() etc.
# But it has evolved, that logit is the easiest to interpret.
 
data = pd.read_csv("~/Code/data/TitanicDataset_Train.csv", na_values=["", " "])

X = data.copy()

# Data Cleaning
X.pop("PassengerId")
X.pop("Cabin")
X.pop("Name")
X.pop("Ticket")

X.Age = X.Age.fillna(X.Age.mean()) # Alternative: X = X.dropna(subset=["Age"], axis="index")
X = X[X.Embarked.notna()] # Alternative: X = X.dropna(subset=["Embarked"], axis="index")

X.Sex = X.Sex.factorize()[0] # Alternative: OrdinalEncoder
X.Embarked = X.Embarked.factorize()[0] # Alternative: OrdinalEncoder



## SKLearn - Useful for getting score etc.
# The result is really horrible though
from sklearn import linear_model as lm

# if you want to add different terms like interactions, you have to
# include the following:
from patsy import dmatrices
y, features = dmatrices('Survived ~ Pclass + Sex + Age + SibSp + Parch + Embarked', X)

# otherwise just simply use the whole feature matrix:
# y = X.pop("Survived")

model = lm.LogisticRegression()
results = model.fit(features,y)
print(model.intercept_)
print(model.coef_)

model.score(features,y) # This is the accuracy, not McFadden Pseudo R2


## statsmodels - Way better if you want to see statistical data
import statsmodels.formula.api as smf
model = smf.logit(formula='Survived ~ Pclass + Age + Sex + SibSp + Parch + C(Embarked)', data=X).fit() # Categorical / factorized Variables must be marked, because they are sometims NOT correctly detected
# if writing Age:C(Sex) it would ONLY use the interaction
print(model.summary())
print(model.prsquared) # McFadden Pseudo R2 ... otherwise it would be model.rsquared

# If doing OLS, one would just use R2 to access the accuracy, or do model.predict(X) and the do a mean_squared_error on the results
# For binary results, we just calculate the "hits" ratio

predictions = model.predict(X)
predictions = [1 if i > 0.5 else 0 for i in predictions]
print("Accuracy: ", 1-(X.Survived != predictions).mean())
