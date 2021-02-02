# Feature-Selection-Using-Genetic-Algorithm
Feature selection is the process of reducing the number of input variables when developing a predictive model and here performed using genertic algorithm on the Boston dataset.
Feature selection are primarly didvided as filter based and wrapper based and this genetic algorithm appraoch comes under the wrapper based feature selection technique.
About Boston dataset: There are 13 independet features and 1 dependent feature and out of these 13 we need to select only the moset descriptive features.

One possible solution

bestf :  -0.6369666437456349

best indipendent features selected by genetic algorithm :  [1 1 0 1 1 1 0 1 1 1 1 1 1]

Code structure:

* boston_feature_selection.py : Contains code function (score function for this case) and runs genetic algorithm to perform feature selection on boston dataset.
* genetic_algorithm.py : Contains code for genetic algorithm 
* genetic_algorithm_utility.py : Contains code for init,neigbour,tournament selction and uniform crossover functions  

Variation of cost function with respect to iteration:

![alt text](https://github.com/prakhargurawa/Feature-Selection-Using-Genetic-Algorithm/blob/main/images/costvsiter.png?raw=true)


Standard deviation of cost with respect to iteration:

![alt text](https://github.com/prakhargurawa/Feature-Selection-Using-Genetic-Algorithm/blob/main/images/stdvsiter.png?raw=true)

