# This confidence interval is for binomial trials
# There must be only two states.

# Otherwise please review Poisson or similar count models

## Source for code: https://de.wikipedia.org/wiki/Konfidenzintervall_f%C3%BCr_die_Erfolgswahrscheinlichkeit_der_Binomialverteilung

from scipy import stats
n = 10000
k = 1000
alpha = 0.05
lower_bound = stats.beta.ppf(alpha/2, k, n-k+1)
upper_bound = stats.beta.ppf(1-alpha/2, k+1, n-k)
print("Measured probability is: %.2f" % (k/n))
print(f"The expected probability is between {lower_bound} and {upper_bound} ")


