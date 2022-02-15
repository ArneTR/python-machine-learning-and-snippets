import pandas as pd
import numpy as np
import scipy
from scipy.stats.morestats import Std_dev


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




## T-Test revalidation if raw numbers are present
#  
# Usually our samples are paired. We measure values on a group. Then we measure the same values on the group after a treatment
# In this case we use the paired T-Test. Recalculate the values and look for methodlogical errors in the study


scipy.stats.ttest_rel(x,y)

# Important: Often a study does not report before and after values and ONLY report the differences. This is tricky as it will skew the T-Test. as we see in the following
# sample, where a is the x is a forced baseline of 1 aka 100% and y is the relative change in percent. 
# If we recalculate with real values (a) and then apply the ratio for the new array and run the t-test again, we get different results!

a = pd.Series([123,178,179,124,144,120,119,119,118,122,126,127,125,130,129,121,120,117,123,126,126,118])
b = a*y

scipy.stats.ttest_rel(x,y)
scipy.stats.ttest_rel(a,b)


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


