import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import numpy as np
import scipy.stats



## Assumptions and prerequisites
#
## ALL of these tests assume real values data. You are NOT allowed to use them
# on ordinal data or bounded data or intervals or categories etc.
# 
## All aof the tests assume i.i.d.
# See chi-squared tests for ordinal data
#



## Post-Hoc calculations.

# If you do not have raw data, we create a dataframe
n = 10
features = ["Blood Pressure", "Resting Heart Rate", "HRV", "Blood sugar", "HbA1c"]
data = pd.DataFrame(np.zeros([n,len(features)]), columns=features)

pd.concat([pd.DataFrame([[1,2,3,4,5]], columns=features), data], ignore_index=True) # run once for every row


# Do you have a dataframe? Then load it here!
data = pd.read_csv("~/Code/data/my_study_data.csv")

# for the example we use prepared list:
x = pd.Series([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
y = pd.Series([1.16,1.3,1.16,1.15,1.2,1.19,1.13,1.11,1.22,1.08,1.14,1.41,1.09,1.11,1.44,1.33,1.24,1.06,1.11,1.11,1.09,1.26])




## Almost ALWAYS the Welch-Test can be used instead of the T-test. Only for samples < 5 the t-test is better
# The Welch-Test does NOT assume same variances.
# However, there is no equivalent for repeated measurements / paired data. Cause here we usually assume same variances


## T-Test with independent / unpaired data
# 
# Assumptions
# - Data is approx normally distributed
# - Data is real valued and not bounded
# - Samples are i.i.d.

value, pvalue = scipy.stats.ttest_ind(x, y, equal_var=False) # done as Welch-Test through equal_var = False
print(value, pvalue)
if pvalue > 0.05:
	print('Samples are likely drawn from the same distributions (fail to reject H0)')
else:
	print('Samples are likely drawn from different distributions (reject H0)')


## T-Test revalidation if raw numbers are present
#  
# Usually our samples are paired. We measure values on a group. Then we measure the same values on the group after a treatment
# In this case we use the paired T-Test. Recalculate the values and look for methodlogical errors in the study

# Important: Often a study does not report before and after values and ONLY report the differences. This is tricky as it will skew the T-Test. as we see in the following
# sample, where a is the x is a forced baseline of 1 aka 100% and y is the relative change in percent. 
# If we recalculate with real values (a) and then apply the ratio for the new array and run the t-test again, we get different results!

a = pd.Series([123,178,179,124,144,120,119,119,118,122,126,127,125,130,129,121,120,117,123,126,126,118])
b = a*y

scipy.stats.ttest_rel(x,y) # Paired!
scipy.stats.ttest_rel(a,b) # Paired!


## Check T-Test with only the aggregated values present.
#
# Given is the data: 
#   baseline 11.1 +/- 3.1
#   after-treatment: 12.7 +/- 3.3

values = []
for i in range(1,1000): 
    baseline = np.random.normal(11.1,3.1,22)
    treatment = np.random.normal(12.7,3.3,22)
    values.append(scipy.stats.ttest_ind(baseline,treatment).pvalue)
values.sort()
print(values)
print(np.mean(values))

## Warning here! The study claimed a p < 0.001 ... However, when I rebuild this data I only get an average p of 0.21 ...
## However always take into account, that the data might be paired and therefore no comparison through this generation of 
## random pairs is allowed! Just changing to ttest_rel is not allowed here. You need two lists with values at same positions 
## indicating the same person before and after!
## 
## The following code shows this:

baseline = np.random.normal(11.1,3.1,22)
treatment = np.random.normal(12.7,3.3,22)
p_before_ind =scipy.stats.ttest_ind(baseline,treatment).pvalue
p_before_rel =scipy.stats.ttest_rel(baseline,treatment).pvalue

baseline.sort()
treatment.sort()
p_after_ind = scipy.stats.ttest_ind(baseline,treatment).pvalue
p_after_rel = scipy.stats.ttest_rel(baseline,treatment).pvalue

print("Independent ... you see SAME values: ", p_before_ind, p_after_ind)
print("Paired ... you see DIFFERENT values: ", p_before_rel, p_after_rel)

## T-Test for paired data
#
# Same assumptions as for T-Test with unpaired data
# Remember, that also in this test the samples must be i.i.d.
# You are just measuring the same instance but in a before / after scenario.
# It is still not allowed that the same measuremt is repeated or also present in the other group.

# Remember that you CANNOT sort the array. This is important for the evaluation process of the paired data
print(scipy.stats.ttest_rel(x,y))

## ANOVA Testing
# Is done if MORE than 2 samples.
#
# This will NOT tell us about a specific pair. It tells us only if ALL are the same, or AT LEAST ONE is different.
# So often times it may be more helpful to test every combination with a T-Test / Welch-Test
#
# Assumptions: 
# - Samples are i.i.d.
# - normally distributed
# - Same variance for all samples
# - Data is NOT-Paired (Independent and NOT repeated measurements)
from numpy.random import seed
from numpy.random import randn
from scipy.stats import f_oneway
# seed the random number generator
seed(1)
# generate three independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 50
data3 = 5 * randn(100) + 52 # This one has a different mean, so we expect to reject H0
# compare samples
stat, p = f_oneway(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
    
    
## ANOVA for paired data / repeated measurements
# 
# https://www.reneshbedre.com/blog/repeated-measure-anova.html

# Assumptions:
# - Check ALL assumptions of ANOVA plus:
# TODO: https://www.reneshbedre.com/blog/repeated-measure-anova.html    
import pingouin as pg
res = pg.rm_anova()

# Alternative: https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html#statsmodels.stats.anova.AnovaRM

    
    
### ANCOVA Testing    
#
# ANCOVA testing is an advancement to T-Test / ANOVA testing.
# It will also test a hypothesis, but similar to linear regression it is
# capable of controllig for a covariate. It can eliminate the effect of 
# a covariate by calculating the effects on the target variable and then
# substracting this effect from the investigated variable.
#
# This can result in higher or lower significance, as the controlled covariate
# may be either an important part for the target variable, and then therefore
# the investigated variable is not as needed anymore to explain the effect.
# Or the result can be higher, as the controlled variable masks the effect
# like some random noise.


# Assumumptions:
# - Check ALL assumptions of ANOVA plus:
# - Same slopes accross different groups in the covariate
# - Homegenity of variances (Levene Test)
# Very good R code for this: https://www.reneshbedre.com/blog/ancova.html
import numpy as np
import pandas as pd; pd.set_option('mode.chained_assignment','raise');

#create data
df = pd.DataFrame({'technique': np.repeat(['A', 'B', 'C'], 5),
                   'current_grade': [67, 88, 75, 77, 85,
                                     92, 69, 77, 74, 88, 
                                     96, 91, 88, 82, 80],
                   'exam_score': [77, 89, 72, 74, 69,
                                  78, 88, 93, 94, 90,
                                  85, 81, 83, 88, 79]})
#view data 
from pingouin import ancova

# here we compare three groups. Technique_A, technique_B and technique_C
# we suspect that the current_grade may  be an important part in the final exam
# score. So we want to control for it to see if the groups really differ when
# ONLY looking at the final exam score (with current grade effects removed)

ancova(data=df, dv='exam_score', covar='current_grade', between='technique')
# Result: p = 0.03155 ... so we reject H0. They are indeed different!


# Checking with normal ANOVA if different (just out of curiosity)
stat, p = f_oneway(df[df.technique == "A"].exam_score, df[df.technique == "B"].exam_score, df[df.technique == "C"].exam_score)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# This value is even higher! So it seems to be the case, that exam_score has an effect.














































