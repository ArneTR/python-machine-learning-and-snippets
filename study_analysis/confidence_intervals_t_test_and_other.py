import numpy as np
from scipy.stats import t

x = np.random.normal(size=100)

m = x.mean()
s = x.std()

dof = len(x)-1
alpha = .05
confidence = 1- alpha

t_crit = np.abs(t.ppf((1-confidence)/2,dof)) # for two sided!

(m-s*t_crit/np.sqrt(len(x)), m+s*t_crit/np.sqrt(len(x)))



## Other Tests than T-test:
# Unchecked. But Bootstrap Method seems to be a good choice
#
# https://machinelearningmastery.com/confidence-intervals-for-machine-learning/