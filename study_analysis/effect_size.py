import numpy as np
from numpy.random import randn
from numpy.random import seed
import pandas as pd; pd.set_option('mode.chained_assignment','raise');

## First of: There is a plethora of methods for measuring the effect size
# Studys often pick the wrong one, so it is very important to recalculate
 
# These two sites give a good Overview: https://www.datanovia.com/en/lessons/t-test-effect-size-using-cohens-d-measure/
# https://researchpy.readthedocs.io/en/latest/ttest_documentation.html

# I have also saved them both in Joplin

## The general rule is: 
# - you should first find out if you have same variances and the data is paired / not-paired (repeated measurements)
# - Then do use all effect size formulas applicable and use the lowest.

x = pd.Series([123,178,179,124,144,120,119,119,118,122,126,127,125,130,129,121,120,117,123,126,126,118]) # sample data control / before treatment
y = x * pd.Series([1.16,1.3,1.16,1.15,1.2,1.19,1.13,1.11,1.22,1.08,1.14,1.41,1.09,1.11,1.44,1.33,1.24,1.06,1.11,1.11,1.09,1.26]) # sample data after treatment / treatment group

# Cohen's d for ONE sample
sample_mean = np.mean(y)
population_mean = 150 # Must be known
sample_stddev = np.std(y, ddof=1)
cohens_d_one_sample =  (sample_mean-population_mean) / sample_stddev

# Cohen's d for independent samples and same variances
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation - Pooled std.dev is a way of calculating the std.dev in an UNBIASED way and is they way to go for std.dev!
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s
 
cohens_d_indepented_same_variance = cohend(y, x)

# This is usually the best. as it also uses pooled variances to be mostly unbiased
cohens_d_independent_same_variance_hedges_correction = ((len(x) + len(y) - 3) / (len(x)+len(y)-2.25)) * cohens_d_indepented_same_variance

cohens_d_welch_test = np.abs(np.mean(x) - np.mean(y)) / np.sqrt( (np.var(x) + np.var(y)) /2 )

print("Independent")
print("cohens_d_indepented_same_variance", cohens_d_indepented_same_variance)
print("cohens_d_independent_same_variance_hedges_correction", cohens_d_independent_same_variance_hedges_correction)
print("cohens_d_welch_test", cohens_d_welch_test)


## Paired Data / Repeated Measurements

differences_group_values = np.subtract(x,y)
std_dev_differences_group_values = np.std(differences_group_values, ddof=1)
cohens_d_paired_data = abs((np.mean(x) - np.mean(y))/std_dev_differences_group_values)
cohens_d_paired_hedges_correction = ((len(x) - 2) / (len(x)-1.25)) * cohens_d_paired_data

differences_group_values = np.subtract(x,y)
std_dev_base_group = np.std(x) # we use ONLY the pre-test group here
glass_delta = np.abs((np.mean(x) - np.mean(y))/std_dev_base_group)
 

print("Repeated measurements")
print("cohens_d_paired_data", cohens_d_paired_data)
print("glass_delta", glass_delta)
print("cohens_d_paired_hedges_correction", cohens_d_paired_hedges_correction)

# There is also a newer cohens_d for paired data: https://www.real-statistics.com/students-t-distribution/paired-sample-t-test/cohens-d-paired-samples/