import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import numpy as np
import scipy
import scipy.stats


# Required sample size calculation
#
# Formular from wikipedia ... but somehow weird results: https://en.wikipedia.org/wiki/Power_of_a_test#A_priori_vs._post_hoc_analysis
expected_effect = 1.6 # defined as avg_treatment-avg_control
alpha = 0.05
beta = 0.2 # this means a "power" of 0.8
population_std_dev = 3.1
( (scipy.stats.norm.ppf(1-alpha/2)+scipy.stats.norm.ppf(beta)) / (expected_effect / population_std_dev) )**2
# WARNING: Unclear if 1-alpha or 1-alpha/2 !!! I found two competing formulas. This formula gives WAY to low values somehow and must be validated!!!!

# This site: https://clincalc.com/Stats/SampleSize.aspx claims the following formula ... which gives good values but differd from wikipedia
( (scipy.stats.norm.ppf(1-alpha/2)+scipy.stats.norm.ppf(beta))**2 * (population_std_dev**2+population_std_dev**2)  ) / (expected_effect**2) 




## Alternative is with using Cohen's d. There are good tables. But also a pyhton class for T-Tests in particular
# IMPORTANT: Modify for the test you want to use
# estimate sample size via power analysis

from statsmodels.stats.power import TTestIndPower, TTestPower
cohens_d_effect_size = 0.5 # calculates as abs(avg_treatment-avg_control) / population_std_dev ### this is different from solely expected effect!
alpha = 0.05
power = 0.8
print('Sample Size for an independent test: %.3f' % TTestIndPower().solve_power(cohens_d_effect_size, power=power, nobs1=None, ratio=1.0, alpha=alpha))
print('Sample Size for a paired test: %.3f' % TTestPower().solve_power(cohens_d_effect_size, power=power, alpha=alpha))
