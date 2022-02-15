# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from study_analysis import normality_tests

## Assumptions
# 
# - The residuals must be normally distributed
# - TODO: More?


## Statsmodels OLS with formula
import statsmodels.formula.api as smf
model = smf.ols(formula="happ ~ hours + I(hours**2)", data=df).fit()
model.summary()

## Validate assumptions: 1.Normality of residuals
normality_tests.run_normality_test(model.resid)






