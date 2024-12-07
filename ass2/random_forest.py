import pandas as pd
import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from regression_tree import *

# references:
# https://medium.com/@byanalytixlabs/random-forest-regression-how-it-helps-in-predictive-analytics-01c31897c1d4
# https://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/ensembles/RandomForests.pdf
# https://www.researchgate.net/figure/Pseudocode-of-random-Forest-classification-23_fig4_362631001
# https://dataaspirant.com/random-forest-algorithm-machine-learing/
# https://www.geeksforgeeks.org/random-forest-regression-in-python/
# https://medium.com/analytics-vidhya/a-super-simple-explanation-to-regression-trees-and-random-forest-regressors-91f27957f688

class RandomForest(BaseEstimator):
    def __init__(self, n_trees=3, bootstrap_size=0.6, max_depth = 10):
        self.n_trees = n_trees
        self.bootstrap_size = bootstrap_size
        self.max_depth = max_depth

    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0])
        # Parse data into np array for consistency
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X)
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        else:
            y = np.asarray(y)
        
        
        # Generate bootstrapped datasets and train trees
        datasets = self.bootstrap(X,y)
        self.trees = []
        for (X_tree, y_tree) in datasets:
            tree = TreeNode(max_depth=self.max_depth)
            tree.train(X_tree, y_tree)
            self.trees.append(tree)

        print(self.trees)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X)

        # Aggregate predictions from all trees for each sample
        return [np.mean([tree.predict(sample) for tree in self.trees]) for sample in X]
    
    # Might not need cause inherited from BaseEstimator?
    # def get_params(self, deep=True):
    #     return {"n_trees": self.n_trees, "bootstrap_size": self.bootstrap_size, "max_depth": self.max_depth}
    
    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self

    # a.k.a. bagging_random_datasets
    def bootstrap(self, X, y):
        datasets = []
        for _ in range(self.n_trees):
            selected = np.random.choice(len(X), size=int(len(X) * self.bootstrap_size), replace=False)   # replace=True means an element can be chosen multiple times.
            datasets.append((X[selected], y[selected]))
        return datasets
