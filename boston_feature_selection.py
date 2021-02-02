# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:59:23 2021

@author: prakh
"""
# import necessary libraries
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
# import genetic algorithm and utility functions
from genetic_algorithm_utility import tournament_select,init,nbr,uniform_crossover
from genetic_algorithm import GA

# import sklearn preloaded boston dataset which has 13 independent variable and 1 dependent variable
X, y = load_boston(return_X_y=True)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
LR = LinearRegression()

# Cost function is same as objective function
def C(x):
    if sum(x) == 0:
        # all zeros means Xtrain_tmp would be empty
        return 1 # a bad value!
    else:
        # the bitstring x chooses the features among the data X.
        # numpy requires x to be bool, not 0/1, for this.
        x = [bool(xi) for xi in x] 
        Xtrain_tmp = Xtrain[:, x]
        Xtest_tmp = Xtest[:, x]
        # fit and score
        LR.fit(Xtrain_tmp, ytrain)
        return -LR.score(Xtest_tmp, ytest) # R^2 -> larger is better, but we minimising

#################################################################
#  FEATURE SELECTION USING GENERIC ALGORITHM ON BOSTON DATASET  #
#################################################################
n = 13
f = C
popsize = 100
ngens = 100
pmut = 0.1
tsize = 2
bestf, best, h = GA(f,
                    lambda: init(n),
                    nbr,
                    uniform_crossover,
                    lambda pop, popfit: tournament_select(pop, popfit, tsize),
                    popsize,
                    ngens,
                    pmut
)

# history format : gen, (gen+1) * len(popfit), np.min(popfit), np.mean(popfit), np.median(popfit), np.max(popfit), np.std(popfit)
h = np.array(h) 

print("bestf : ",bestf)
print("best indipendent features selected by genetic algorithm : ",best)

"""
One possible solution
bestf :  -0.6369666437456349
best indipendent features selected by genetic algorithm :  [1 1 0 1 1 1 0 1 1 1 1 1 1]
"""

# plot min fit against number of individuals so far. we see fast
# improvements, then a sense of "plateau".
plt.plot(h[:, 1], h[:, 2])
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()
plt.close()

# plot std fit against number of individuals so far.  we see that the
# SD becomes very small after only about 1/5th of the run... but it
# remains non-zero, so the population is not fully converged.
plt.plot(h[:, 1], h[:, -1])
plt.xlabel("Iterations")
plt.ylabel("SD(Cost)")
plt.show()
plt.close()