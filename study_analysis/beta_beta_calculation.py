import pandas as pd
import numpy as np
import scipy

## Power calculation
# Reminder: NEVER do a power calculation afterwards with the "measured" effect. This value has no meaning! See: https://en.wikipedia.org/wiki/Power_of_a_test#A_priori_vs._post_hoc_analysis


alpha = 0.05
n = 40

z = scipy.stats.norm.ppf(alpha) # Value for left-tailed Z-value given alpha = 0.05
z = scipy.stats.t.ppf(alpha,n-1) # Value for left-tailed T-Value given alpha = 0.05
# ...

z = -scipy.stats.norm.ppf(alpha) # Value for right-tailed Z-value given alpha = 0.05
z = -scipy.stats.t.ppf(alpha,n-1) # Value for left-tailed T-Value given alpha = 0.05
# ...

z = scipy.stats.norm.ppf(alpha/2) # Value for two sided Z-value given alpha = 0.05 ... this value must be used as +/- later
z = scipy.stats.t.ppf(alpha/2) # Value for two sided t-value given alpha = 0.05 ... this value must be used as +/- later
# ...


## calculation for left-tailed. Here expected_mean_value must be LESS than current_mean_value
# to calculate the cutoff value where we fail to reject we rearrange the formula z = (X_bar - mu) / (std / np.sqrt(n))
z = scipy.stats.norm.ppf(alpha) # Value for left-tailed Z-value given alpha = 0.05
current_mean_value = 500
std = 24 # this usually is the std of the population. If unknown the sample std should be calculated according to (1/n-1) * SUM(mu-value) (See Joplin B.Statistics MitX)
X_bar_cutoff = current_mean_value + z*(std / np.sqrt(n)) # This is the value we cutoff for a X-tailed test. It is either higher or lower than H0

expected_mean_value = 490  # this is the value we hope that our drug / medicamentation can achieve AT LEAST
beta = 1-scipy.stats.norm.cdf((X_bar_cutoff - expected_mean_value) / (std / np.sqrt(n)))
power = 1-beta

# calculation for right-tailed
# to calculate the cutoff value where we fail to reject we rearrange the formula z = (X_bar - mu) / (std / np.sqrt(n))
z = -scipy.stats.norm.ppf(alpha) # Value for left-tailed Z-value given alpha = 0.05
current_mean_value = 500
std = 24 # this usually is the std of the population. If unknown the sample std should be calculated by np.std(array_like, ddof=1)
X_bar_cutoff = current_mean_value + z*(std / np.sqrt(n)) # This is the value we cutoff for a X-tailed test. It is either higher or lower than H0

expected_mean_value = 510  # this is the value we hope that our drug / medicamentation can achieve AT LEAST
beta = scipy.stats.norm.cdf((X_bar_cutoff - expected_mean_value) / (std / np.sqrt(n)))
power = 1-beta