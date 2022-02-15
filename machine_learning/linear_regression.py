import statsmodels.formula.api as smf
import numpy as np
import scipy.stats as stats

#add legend
hours = [6, 9, 12, 12, 15, 21, 24, 24, 27, 30, 36, 39, 45, 48, 57, 60]
happ = [12, 18, 30, 42, 48, 78, 90, 96, 96, 90, 84, 78, 66, 54, 36, 24]



print(smf.ols("hours ~ happ", {"hours": hours, "happ": happ}))











