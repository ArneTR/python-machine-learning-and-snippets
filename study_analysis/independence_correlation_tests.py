
####### Example of the Pearson's Correlation test
#Tests whether two samples have a linear relationship.

#Assumptions

#Observations in each sample are independent and identically distributed (iid).
#Observations in each sample are normally distributed.
#Observations in each sample have the same variance.
#Interpretation

#H0: the two samples are independent.
#H1: there is a dependency between the samples.

from scipy.stats import pearsonr
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')




##### Spearman’s Rank Correlation
#Tests whether two samples have a monotonic relationship.
# They do NOT need to be normally distributed

#Assumptions
#Observations in each sample are independent and identically distributed (iid).
#Observations in each sample can be ranked.

#Interpretation
#H0: the two samples are independent.
#H1: there is a dependency between the samples.

from scipy.stats import spearmanr
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')    



######## Chi-Squared Test
# Tests whether two categorical variables are related or independent.

# Assumptions

# Observations used in the calculation of the contingency table are independent.
# 25 or more examples in each cell of the contingency table.

#Interpretation
#H0: the two samples are independent.
#H1: there is a dependency between the samples.

from scipy.stats import chi2_contingency
table = [[10, 20, 30],[6,  9,  17]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')	

	# chi-squared test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2



# chi-squared test with similar proportions
from scipy.stats import chi2_contingency
from scipy.stats import chi2
# contingency table
table = [	[10, 20, 30],
			[6,  9,  17]]
print(table)
stat, p, dof, expected = chi2_contingency(table)
print('dof=%d' % dof)
print(expected)

alpha = 0.05
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')

## Another more uncommon way, that leads to the same result:
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
	