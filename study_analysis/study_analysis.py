import pandas as pd
import numpy as np
import scipy
from scipy.stats.morestats import Std_dev

## first we talk about some easiy calculations we do BEFORE looking at the gathered data.
# Here we usually check assumptions and modelling and if they are valid.
# This includes power calculation, sample size calculation etc.

## Here we to calculate a type_I error based on a ALREADY set cutoff value  (assumption is we are dealing with normal distribution. No samples)
# Sometimes cutoff values are known before. But this is realyy seldom, so this calculation is most often not used

# If the cholesterol level of healthy men is normally distributed with a mean of 180 and a standard deviation of 20,
#  and men with cholesterol levels over 225 are diagnosed as not healthy, what is the probability of a type one error?
population_mean = 180
population_std_dev = 20
measured_value = 225
1-scipy.stats.norm.cdf((measured_value-population_mean) / population_std_dev) # probability_of_type_I_error

## Here we calculate the cutoff value based on a WANTED type I error (assumption is we are dealing with normal distribution. No samples)
# This is usually done to help with a wanted effect size assumption. For power-testing we NEED an estimate / wish for the effect.

#If the cholesterol level of healthy men is normally distributed with a mean of 180 and a standard deviation of 20, 
# at what level (in excess of 180) should men be diagnosed as not healthy if you want the probability of a type one error to be 0.02?
wanted_type_I_error = 0.02
scipy.stats.norm.ppf(1-wanted_type_I_error) * population_std_dev + population_mean # cutoff value for results, that are considered significant

## Power calculation
# Reminder: NEVER do a power calculation afterwards with the "measured" effect. This value has no meaning! See: https://en.wikipedia.org/wiki/Power_of_a_test#A_priori_vs._post_hoc_analysis


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
effect = 0.5 # calculates as abs(avg_treatment-avg_control) / population_std_dev ### this is different from solely expected effect!
alpha = 0.05
power = 0.8
print('Sample Size for an independent test: %.3f' % TTestIndPower().solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha))
print('Sample Size for a paired test: %.3f' % TTestPower().solve_power(effect, power=power, alpha=alpha))

######### DEFINTION of Alpha ##############
# If there were actually no effect (if the true difference between means were zero) then the probability 
# of observing a value for the difference equal to, or greater than, that actually observed would be p=0.05. I
# n other words there is a 5% chance of seeing a difference at least as big as we have done, by chance alone.


# Was mich halt richtig stört: Es gibt widersprüchliche Aussage zur Power Calculation. Einmal geht es gar nicht für Hyptohesen wo H1 > y ist. 
# Dann ist es in anderen Rechnern wieder kein PRoblem. Oft gibt es viele Rechnenwege für das gleiche.
# SUPER nervig.
# Dann auch auch diese arbiträren Festlegungen: Cohen sagt, dass die Power von 0.8 ein hoher Wert ist. TOLL! Vielen Dank!
#
# Dann auch diese Annahmen die immer getroffen sein müssen: Muss eine Zufallsgezogene Stichprobe sein, der Mittelwert muss geschätzt werden, die Varianz muss geschätzt werden
# usw. ... iwo macht man immer Fehler ... und selbst wenn nicht sind die Rechnenvorschriften nur "Vorschläge", keine logischen beweisbaren Sachen. KOTZ!
#
# In dem Artikel https://royalsocietypublishing.org/doi/10.1098/rsos.140216#d3e749 sieht man schön, dass man das Konzept beliebig verbiegen kann.
# WEnn man den p-Wert von verschiendenen Standpunkten betrachtet oder das Experiment wiederholt, dann kommt nur Murks raus.
# 
# Schlussendlich bleiben einfache Gesetze: Man wiederhole eine Experiment 5 mal und gucke wie oft das gleiche Ergebnis rauskommt. Fertig.
# Der Wahnsinn aus einem Experiment eine Aussage treffen zu können und dann oft auch noch das Minimum an Probanden zu nehmen ist einfach mathematisch kaum zu halten.
# Ein einfacher Test mit p=0.05 hat lediglich die Aussagekraft: Kann man sich mal näher angucken. Aber bei weitem nicht mehr.

# WEiterhin sind folgende Regeln gut, falls es nicht möglich ist eine Studie zu wiederholen: Sample Size Calculations + p=0.001
 

## Post-Hoc calculations.

# If you do not have raw data, we create a dataframe
n = 10
features = ["Blood Pressure", "Resting Heart Rate", "HRV", "Blood sugar", "HbA1c"]
data = pd.DataFrame(np.zeros([n,len(features)]), columns=features)

pd.concat([pd.DataFrame([[1,2,3,4,5]], columns=features), data], ignore_index=True) # run once for every row


# Do you have a dataframe? Then load it here!
data = pd.read_csv("~/Code/data/my_study_data.csv")


## T-Test revalidation if raw numbers are present
#  
# Usually our samples are paired. We measure values on a group. Then we measure the same values on the group after a treatment
# In this case we use the paired T-Test. Recalculate the values and look for methodlogical errors in the study

x = pd.Series([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
y = pd.Series([1.16,1.3,1.16,1.15,1.2,1.19,1.13,1.11,1.22,1.08,1.14,1.41,1.09,1.11,1.44,1.33,1.24,1.06,1.11,1.11,1.09,1.26])

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
treatment = np.random.normal(12.7,3.3,22)p_before_ind =scipy.stats.ttest_ind(baseline,treatment).pvalue
p_before_rel =scipy.stats.ttest_rel(baseline,treatment).pvalue

baseline.sort()
treatment.sort()
p_after_ind = scipy.stats.ttest_ind(baseline,treatment).pvalue
p_after_rel = scipy.stats.ttest_rel(baseline,treatment).pvalue

print("Independent ... you see SAME values: ", p_before_ind, p_after_ind)
print("Paired ... you see DIFFERENT values: ", p_before_rel, p_after_rel)


## Test for similarity to distribution / normality

# Version 1: QQPlot. Should follow 45 degrees line. Can check any distribution

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

data_points = np.random.normal(0, 1, 100) # in the sample we use normal distribution   
sm.qqplot(y, dist=scipy.stats.norm, line="45") 
plt.show()

# however for real data, you must shift the comparing normal distribution
y = pd.Series([1.16,1.3,1.16,1.15,1.2,1.19,1.13,1.11,1.22,1.08,1.14,1.41,1.09,1.11,1.44,1.33,1.24,1.06,1.11,1.11,1.09,1.26])
sm.qqplot(y, loc=np.mean(y), scale=np.std(y), line="45")
plt.show()


## Version 2 KS.Test
from scipy.stats import kstest
import random
    
kstest(y, y) # test with another drawn distribution   
statistics, p = kstest(y, scipy.stats.norm.rvs(loc=np.mean(y), scale=np.std(y), size=10000)) # Test against normal distribution
if p > 0.05: 
    print("According to KS-Test sample looks gaussian")
else:
    print("According to KS-Test sample does NOT looks gaussian")

# Downsides: The test has a low "Sensitivität". Meaning that it will quite seldom tell, that the distributions are different. Even if they are.
# However, when the test tells, that they are diffrent (In R through a p-value below 5% or 1% etc.) than it is highly likely not the same distribution.
# So the test is a first indicator! And if we have two samples we might suggest, that they are really different.

## Version 3: Shapiro Test
stat, p = scipy.stats.shapiro(np.random.normal(0,1,10)) # this will be normal
stat, p = scipy.stats.shapiro(y) # this is our demo sample
print(stat, p)
if p > 0.05: 
    print("According to Shapiro sample looks gaussian")
else:
    print("According to Shapiro sample does NOT looks gaussian")





### TEST AREA
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
# to calculate the cutoff value where we fail to reject we rearrange the formula z = (X_bar - mu) / (std_expected**2 / n)
z = scipy.stats.norm.ppf(alpha) # Value for left-tailed Z-value given alpha = 0.05
current_mean_value = 500
std_expected = 24
X_bar_cutoff = current_mean_value + z*(std_expected / np.sqrt(n)) # This is the value we cutoff for a X-tailed test. It is either higher or lower than H0

expected_mean_value = 490  # this is the value we hope that our drug / medicamentation can achieve AT LEAST
1-scipy.stats.norm.cdf((X_bar_cutoff - expected_mean_value) / (std_expected / np.sqrt(n)))


# calculation for right-tailed
# to calculate the cutoff value where we fail to reject we rearrange the formula z = (X_bar - mu) / (std_expected**2 / n)
z = -scipy.stats.norm.ppf(alpha) # Value for left-tailed Z-value given alpha = 0.05
current_mean_value = 500
std_expected = 24
X_bar_cutoff = current_mean_value + z*(std_expected / np.sqrt(n)) # This is the value we cutoff for a X-tailed test. It is either higher or lower than H0

expected_mean_value = 510  # this is the value we hope that our drug / medicamentation can achieve AT LEAST
scipy.stats.norm.cdf((X_bar_cutoff - expected_mean_value) / (std_expected / np.sqrt(n)))
