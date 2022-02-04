import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import numpy as np
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt


def run_normality_test(y = None):
    if(y is None):
        # our sample data
        y= pd.Series([1.16,1.3,1.16,1.15,1.2,1.19,1.13,1.11,1.22,1.08,1.14,1.41,1.09,1.11,1.44,1.33,1.24,1.06,1.11,1.11,1.09,1.26])
    elif(type(y) != pd.Series):
        y = pd.Series(y)
    ## Test for similarity to distribution / normality
    # 
    # Always remember: If the checks are not giving good results you are allowed
    # and encouraged to transform the data
    
    
    
    # Version 0: The histogram first look
    y.hist()
    
    
    # Version 1: QQPlot. Should follow 45 degrees line. Can check any distribution    
    # however for real data, you must shift the comparing normal distribution
    sm.qqplot(y, loc=np.mean(y), scale=np.std(y), dist=scipy.stats.norm, line="45")
    plt.show()
    
    
    ## Version 2 KS.Test
    from scipy.stats import kstest
        
    kstest(y, y) # test with another drawn distribution   
    statistics, p = kstest(y, scipy.stats.norm.rvs(loc=np.mean(y), scale=np.std(y), size=10000)) # Test against normal distribution
    if p > 0.05: 
        print("According to KS-Test sample looks gaussian, alpha 0.05")
    else:
        print("According to KS-Test sample does NOT looks gaussian, alpha 0.05")
    
    # Downsides: The test has a low "Sensitivität". Meaning that it will quite seldom tell, that the distributions are different. Even if they are.
    # However, when the test tells, that they are diffrent (In R through a p-value below 5% or 1% etc.) than it is highly likely not the same distribution.
    # So the test is a first indicator! And if we have two samples we might suggest, that they are really different.
    
    ## Version 3: Shapiro Test
    stat, p = scipy.stats.shapiro(np.random.normal(0,1,10)) # this will be normal
    stat, p = scipy.stats.shapiro(y) # this is our demo sample
    print(stat, p)
    if p > 0.05: 
        print("According to Shapiro sample looks gaussian, alpha 0.05")
    else:
        print("According to Shapiro sample does NOT looks gaussian, alpha 0.05")
    
    
    # Version 4: D'Agostino K^2 Test
    from scipy.stats import normaltest
    stat, p = normaltest(y)
    alpha = 0.05
    if p > alpha:
    	print('According to D\'Agostino sample looks Gaussian (fail to reject H0), alpha 0.05')
    else:
    	print('According to D\'Agostino sample not look Gaussian (reject H0), alpha 0.05')
        
    # There are also many more test: https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    # Generally one should to as many tests as possible.
    # If you MUST be extremely normal, then if one tests fails, you cannot assume gaussian style
    # However often in machine learning and regression you do not have to be 100% gaussian.
    # a gaussian-like is often ok. So having only some tests fail (less than 50% of the tests!) may be sufficient



        
        
        
        
        