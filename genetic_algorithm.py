# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:57:22 2021

@author: prakh
"""

# import necessary libraries
import numpy as np
import random
import matplotlib.pyplot as plt

# GA for minimisation
def GA(f, init, nbr, crossover, select, popsize, ngens, pmut):
    history = []
    # make initial population, evaluate fitness, print stats
    pop = [init() for _ in range(popsize)]
    popfit = [f(x) for x in pop]
    history.append(stats(0, popfit))
    for gen in range(1, ngens):
        # make an empty new population
        newpop = []
        # elitism : directly select the best candidate to next population as it is
        bestidx = min(range(popsize), key=lambda i: popfit[i])
        best = pop[bestidx]
        newpop.append(best)
        while len(newpop) < popsize:
            # select and crossover
            p1 = select(pop, popfit)
            p2 = select(pop, popfit)
            c1, c2 = crossover(p1, p2)
            # apply mutation to only a fraction of individuals : pmut is hyperparameter
            if random.random() < pmut:
                c1 = nbr(c1)
            if random.random() < pmut:
                c2 = nbr(c2)
            # add the new individuals to the population
            newpop.append(c1)
            # ensure we don't make newpop of size (popsize+1) - 
            # elitism could cause this since it copies 1
            if len(newpop) < popsize:
                newpop.append(c2)
        # overwrite old population with new, evaluate, do stats
        pop = newpop
        popfit = [f(x) for x in pop]
        history.append(stats(gen, popfit))
    bestidx = np.argmin(popfit)
    return popfit[bestidx], pop[bestidx], history

def stats(gen, popfit):
    # let's return the generation number and the number
    # of individuals which have been evaluated
    return gen, (gen+1) * len(popfit), np.min(popfit), np.mean(popfit), np.median(popfit), np.max(popfit), np.std(popfit)