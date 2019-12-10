#!/usr/bin/env python
# coding: utf-8

# # Assignment 3: Confidence Intervals & The Bootstrap
# 
# ### Instructions
# 
# This assignment is much like the last.
# 
# * This assignment includes some tests to help you make sure that your implementation is correct.  When you see a cell with `assert` commands, these are tests.
# 
# * Once you have completed the assignment, delete the cells which have these `assert` commands.  You will not need them.
# 
# * When you are done and have answered all the questions, convert this notebook to a .py file using `File > Download as > Python (.py)`.  Name your submission `assignment3.py` and submit it to OWL.
# 
# Failure to comply may resilt in you not earning full marks for your assignment.  We want you to earn full marks!  Please follow these instructions carefully.

# # Question 1
# 
# ### Part A
# 
# Recall from the lecture that a $100(1-\alpha)\%$ confidence interval for the mean is 
# 
# $$ \bar{x} \pm  t_{1-\alpha/2, n-1} \dfrac{\hat{\sigma}}{\sqrt{n}} $$
# 
# Where $ t_{1-\alpha/2, n-1}$ is the appropiorate quantile of a Student's t distribution with $n-1$ degrees of freedom.  When $\alpha = 0.05$ and when $n$ is big enough, $ t_{1-\alpha/2, n-1} \approx 1.96$.  
# 
# Write a function called `confidence_interval` which takes as it's argument an array of data called `data` and returns two things:
# 
# * An estimated mean of `data`, and 
# 
# * The lower and upper bounds of the 95% confidence interval for the mean of `data`.  Ensure these are returned in a numpy array of shape (2,)

# In[56]:


#It's dangerous to go alone.  Take these
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import pandas as pd
import seaborn as sns


def confidence_interval(data):
    '''
    Function to compute confidence interval (ci).
    
    Inputs:
    data - ndarray.  Data to be used to compute the interval
    
    Outputs:
    estimated_mean - float.  The mean of the data
    bounds - array. An array of length 2 where bounds[0] is the lower bound and bounds[1] is the upper bound for the ci.
    '''
    ### BEGIN SOLUTION
    estimated_mean = data.mean()
    df = len(data) - 1
    critval = t.ppf(0.975, df=df)
    stderr = np.std(data)/np.sqrt(len(data))
    bounds = np.array([data.mean() - stderr*critval,
           data.mean() + stderr*critval])
    
    
    ### END SOLUTION
    
    return estimated_mean, bounds


# In[3]:


confidence_interval(np.array([-1,0,1]))


# ### Tests
# 
# Tests are to make sure you've implemented the solution correctly.  If these tests pass without any `AsserstionError`'s, then you can be confident that you've implemented the solution as expected.
# 
# Once you're happy with your implementation, delete the cell below.

# ### Part B
# 
# The "95% confidence interval" is named so because the long term relative frequency of these estimtors containing the true estimand is 95%.  That is to say **if I construct 95% confidence intervals for the sample mean again and again from the same data generating mechanism, 95% of the intervals I construct will contain the population mean**.
# 
# Write a function called `ci_simulation` that runs some simulations to show this is the case.  From a standard normal distirbution, sample 25 observations and construct a confidence interval.  Do this 20 times and plot the intervals using `matplotlib.pyplot.errorbar`. Save your plot under the name `ci_simulation.png`.  Color the bar red if the confidence interval does not caputre the true mean and blue if it does.  If you are unfamilliar with `matplotlib.pyplot.errorbar`, I highly suggest reading Matplotlib's excellent documentation.

# In[5]:


def ci_simulation():
    # Reproducibility.  Do not change this!
    np.random.seed(3)
    
    # Create the figure.
    fig, ax = plt.subplots(dpi = 120)

    # If the interval crosses this line, it should be blue, else red.
    ax.axhline(0, color = 'k')
    
    ### BEGIN SOLUTION
    mu, sigma = 0, 1
    n = 25
    
    for i in range(20):
        data = np.random.normal(mu, sigma, n)#, replace=True) # sample n obs. from a normal distribution
        _, ci = confidence_interval(data) # compute CI for sample
        if ci[0] > 0 or ci[1] < 0:
            plt.errorbar(i, data.mean(), yerr=(ci[1]-ci[0])/2, ecolor='r',
                        fmt='o-r')
        else:
            plt.errorbar(i, data.mean(), yerr=(ci[1]-ci[0])/2, ecolor='b',
                        fmt='o-b')
        
        
    ### END SOLUTION
    # This function does not have to return anything
    return None

ci_simulation()


# ### Part C
# 
# If you haven't changed the random seed from 3 and if you implemented the solution correctly, you should two red intervals.
# 
# Answer the following below in 1-2 sentences:
# 
# 1) How many red intervals did we expect to see?  What is your justifiation for this?
# 
# 2) If there is a discrepency between the number of observed and expected red intervals, what explains this difference?
# 
# ### Answers
# 
# 1) We expected 1 interval. 5% of 20 is 1, so we expect 1 of the intervals to be wrong.
# 
# 2) Random sampling.

# ### Part D
# 
# How many samples would we need in order to ensure that our constructed confidence interval is approximately 0.1 units long? 
# 
# Write a function called `num_propper_length` which takes as its only argument an integer `n`.  `num_propper_length` should simulate 1000 datasets of length `n`, compute confidence intervals for those datasets, compute the lengths of those intervals, and then returns the number of intervals which are no longer than 0.1 units.
# 
# Determine how many samples you need (that is, compute `n`).  Set this as your default argument for `n` in `num_propper_length`.

# In[7]:


def num_propper_length(n=1600):
    '''
    Function to simulate how many out of 1000 confidence intervals
   would be no longer than 0.1 units long if
    we sampled n observations from a standard normal.
    
    Inputs:
        n - int.  Number of draws to make from the standard normal
        
    Outputs:
        num_long_enough - integer.  Number of constructed intervals which are no longer than 0.1.
    '''
    # For reproducibility.  Don't change this!
    np.random.seed(0)
    
    ### BEGIN SOLUTION
    num_long_enough = 0
    for i in range(1000):
        data = np.random.normal(0, 1, n)#, replace=True) # sample n obs. from a normal distribution
        _, ci = confidence_interval(data)
        if ci[1] - ci[0] <= 0.1:
            num_long_enough += 1

    ### END SOLUTION
    return num_long_enough

num_propper_length(n=1600)


# ### Tests
# 
# Tests are to make sure you've implemented the solution correctly.  If these tests pass without any `AsserstionError`'s, then you can be confident that you've implemented the solution as expected.
# 
# Once you're happy with your implementation, delete the cell below.

# ### Part E
# If you chose the right `n`, you should find that 891 (or approximately 89%) of your intervals are longer than 0.1.  
# 
# Why is this?  Answer below in 1-2 sentences.
# 
# ### Answer:
# 
# Random sampling.

# ---

# ## Question 2
# 
# ### Part A
# The dataset `hockey_stats.csv` contains information about information about hockey draftees.  We'll use this data to investigate the relationship between height and age on weight.  Load it into python using pandas.
# 
# Load in the `hockey_draftees_train.csv` data into pandas.  Fit a linear model of weight (`wt`) explained by height (`ht`) and age(`age`).  Call your fitted model `model`.

# In[28]:


### BEGIN SOLUTION
data = pd.read_csv('hockey_draftees_train.csv')

model = sm.ols('wt ~ ht + age', data=data).fit()

model.summary()

### END SOLUTION


# ### Part B
# 
# Print out the R-squared for this model
# 

# In[31]:


print('Training rsquared is ',model.rsquared)
# 31.4% of variation in weight is explained by height and age.


# ### Part C
# 
# Now, let's see how well our model performs out of sample.  Load in the `hockey_draftees_test.csv` file into a dataframe.  Use your `model` to make predictions, and print the r-squared on the out of sample (oos) data.

# In[51]:


### BEGIN SOLUTION
test = pd.read_csv('hockey_draftees_test.csv')

y = test['wt']

X = test[['ht', 'age']]

yhat = model.predict(X)

yhatmew = yhat.mean()

yhatmew = yhatmew*np.ones(len(y))

SSE = sum((y.values - yhat)**2)

SST = sum((y.values - yhatmew)**2)

rsquared_oos = 1 - SSE/SST

### END SOLUTION

print('Out of sample rsquared is ', rsquared_oos)


# ### Part D
# 
# A point estimate of the rsquared is nice, but what we really want is uncertainty estimates.  For that, we need a confidence interval.  To estimate how uncertain we are in our out of sample r-squared, let's use the bootstrap.
# 
# Write a function called `bootstrap` which takes three arguments:
# 
# * `data`, which is a dataframe, and 
# * `model` which is an statsmodel ols model. `data` should look the the data `model` was trained on so that we can use `model` to make predictions on `data`.
# * `numboot` which is an integer denoting how many bootstrap replications to perform.
# 
# Write `bootstrap` to perform bootstrap resampling for the out of sample r-squared.  You can use `pd.DataFrame.sample` with `replace = True` to perform the resampling.
# 
# `bootstrap` should return a numpy array of bootstraped rsquared values.
# 
# 

# In[67]:


def bootstrap(data, model, numboot):
    '''
    Function to bootstrap the r-squared for a linear model
    
    Inputs:
        data - dataframe.  Data on which model can predict.
        model - statsmodel ols.  Linear model of weight explained by height and age.  Can predict on data.
        numboot - int.  Number of bootstrap replications.
    
    Outputs:
        bootstrapped_rsquared - array.  An array of size (numboot, ) which contains oos bootstrapped rsquared values.
    
    '''
    bootstrapped_rsquared = np.ones(numboot)
    ### BEGIN SOLUTION
    np.random.seed(3)
    for i in range(numboot):
        n = len(data)
        
        d = data.sample(n, replace=True)
        
        y = d['wt']

        X = d[['ht', 'age']]

        yhat = model.predict(X)

        yhatmew = yhat.mean()

        yhatmew = yhatmew*np.ones(len(y))

        SSE = sum((y.values - yhat)**2)
    
        SST = sum((y.values - yhatmew)**2)

        rsquared_oos = 1 - SSE/SST
        
        bootstrapped_rsquared[i] = rsquared_oos
        
    
    ### END SOLUTION

    return bootstrapped_rsquared


# ### Part E
# 
# Use your `bootstrap` function to plot 10,000 bootstrap replicates as a histogram.

# In[68]:


### BEGIN SOLUTION
bs = bootstrap(test, model, 10000)
### END SOLUTION


# In[69]:


sns.distplot(bs)


# ### Part F
# 
# Use your bootstrap replicates to estimates to obtain a bootstrapped 95% confidence interval.  Call the upper confidence bound `ci_upper` and the lower confidence bound `ci_lower`.

# In[70]:


### BEGIN SOLUTIOn
boot_ci = np.quantile(bs, [0.025, 0.975])
ci_lower = boot_ci[0]
ci_upper= boot_ci[1]
### END SOLUTION

print('My confidence interval is between', ci_lower, ' and ', ci_upper)

