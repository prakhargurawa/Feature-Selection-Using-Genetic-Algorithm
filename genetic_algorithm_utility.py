# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:58:22 2021

@author: prakh
"""

# import necessary libraries
import random
import numpy as np

def tournament_select(pop, popfit, size):
    # To avoid re-calculating f for same individual multiple times, we
    # put fitness evaluation in the main loop and store the result in
    # popfit. We pass that in here.  Now the candidates are just
    # indices, representing individuals in the population.
    candidates = random.sample(list(range(len(pop))), size)
    # The winner is the index of the individual with min fitness.
    winner = min(candidates, key=lambda c: popfit[c])
    return pop[winner]

# for feature selection problem set we need to have binary init and neighbour functions

# our standard bitstring init. use lambda: init(n) to give a function
# that takes no parameters.
def init(n):
    return [random.randrange(2) for _ in range(n)]

# our usual bitstring nbr
def nbr(x):
    x = x.copy()
    i = random.randrange(len(x))
    x[i] = 1 - x[i]
    return x

# we have used uniform crossover, other option one point and multiple point crossovers
def uniform_crossover(p1, p2):
    c1, c2 = [], []
    for i in range(len(p1)):
        if random.random() < 0.5:
            c1.append(p1[i]); c2.append(p2[i])
        else:
            c1.append(p2[i]); c2.append(p1[i])
    return np.array(c1), np.array(c2)