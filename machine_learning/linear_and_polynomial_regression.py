import numpy as np
import scipy.stats as stats
import pandas as pd


## Assumptions
# 
# - The residuals must be normally distributed
# TODO: How do I get the residuals out of statsmodels?

#add legend
hours = pd.Series([6, 9, 12, 12, 15, 21, 24, 24, 27, 30, 36, 39, 45, 48, 57, 60])
happ = pd.Series([12, 18, 30, 42, 48, 78, 90, 96, 96, 90, 84, 78, 66, 54, 36, 24])

import matplotlib.pyplot as plt

#create scatterplot
plt.scatter(hours, happ)
plt.show()

## classical polynomial model with numpy
np_model = np.poly1d(np.polyfit(hours, happ, 2)) #polynomial fit with degree = 2


#add fitted polynomial line to scatterplot
polyline = np.linspace(1, 60, 50)
plt.scatter(hours, happ)
plt.plot(polyline, model(polyline))
plt.show()

np_model # -0.107x2 + 7.173x - 30.25


import statsmodels.formula.api as smf
df = pd.DataFrame({"hours": hours, "happ": happ})

## This model is way easire to read and cann inject polynomial relations directly
# See the coefficients to notice that it is the exact same model
model = smf.ols(formula="happ ~ hours + I(hours**2)", data=df).fit()

## Validate assumptions:
# Normality of residuals
from study_analysis import normality_tests
normality_tests.run_normality_test(model.resid)


# Assumption #2: TODO

# If variables are categorical, the can be easily spread out
smf.ols(formula="happ ~ hours + C(hours)", data=df).fit().summary() 





