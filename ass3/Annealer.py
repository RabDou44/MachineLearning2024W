import random

import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import itertools
import pandas as pd
import time

class Annealer:
    """
    Annealer class that performs grid search on a given method
    Parameters:
    -----------
    method: sklearn estimator
    feature_structure: dict containing the feature structure of the data
    search_space: dict - parameters value
    max_iter: int - maximum number of iterations
    metric: str - metric to use for evaluation
    data: pandas DataFrame - dataset
    """
    required_keys = {'bin', 'cat', 'cont', 'ord', 'target'}

    def __init__(self, method = BaseEstimator(),
                 feature_structure = {},
                 search_space = {},
                 max_iter = 100,
                 metric = 'accuracy',
                 data = None,
                 fold_num = 5):

        assert isinstance(feature_structure, dict),"Feature_structure should be a dictionary"
        assert isinstance(search_space, dict), "Search_space should be a dictionary"
        assert isinstance(method, BaseEstimator), "Method should be a sklearn estimator"
        assert self.required_keys.issubset(feature_structure.keys()), "Feature structure must contain the keys:  {'bin', 'cat', 'cont', 'ord', 'target'}"
        self.__method__  = method
        self.__search_spaces__ = search_space
        self.__max_iter__ = max_iter
        self.__metric__ = metric
        self.__fold_num__ = fold_num
        self.G = None

        ## predefine preprocessing steps
        categorical_preprocessor = Pipeline(
            steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

        numerical_preprocessor = Pipeline([
            ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ('scaler', RobustScaler())
        ])

        ordinal_preprocessor = Pipeline([
            ("ordinal", OrdinalEncoder()),
        ])

        self.preprocessor_step = ColumnTransformer([
            ('categorical', categorical_preprocessor, feature_structure['cat']),
            ('numerical', numerical_preprocessor, feature_structure['cont']),
            ('ordinal', ordinal_preprocessor, feature_structure['ord'])
        ])

        ## split dataset
        self.X = data.drop(columns=feature_structure['target']).reset_index(drop=True)
        self.y = data[feature_structure['target']].reset_index(drop=True)

    def grid_search(self):
        """
        Perform grid search on the given method
        """
        # currently typical grid search
        # assert(self.check_param_space(), "The search space is not a subset of the method's parameters")
        # assert(self.__feature_structure__,"Feature structure is not defined"
        param_grid = pd.DataFrame(self.get_full_grid(), columns=self.__search_spaces__.keys())
        best_params = None
        best_score = - np.inf
        timing = 0

        for index, params in param_grid.iterrows():

            if index >= self.__max_iter__:
                break

            dict_params =  dict(zip(param_grid.keys(), params))
            model = Pipeline(steps=[('preprocessor', self.preprocessor_step), ('classifier', self.__method__.set_params(**dict_params))])

            # cross validation
            start_time = time.time()
            score = cross_val_score(model, self.X, self.y, cv=self.__fold_num__, scoring=self.__metric__, n_jobs =-1).mean()
            timing += time.time() - start_time

            if score > best_score:
                best_score = score
                best_params = dict_params

            print(f"Iteration score {index+1}: {score} with params: {dict_params}")

        print(f"""Best score: {best_score} with params: {best_params}""")
        return best_params, best_score, timing

    def hill_climbing(self, curr_node=None, max_iter=100):
        # build search graph
        start_time = time.time()
        G = self.G
        if self.G is None:
            G =  self.build_graph_search_space().copy()
        else:
            G = self.G.copy()

        visited_nodes = set()
        candidates = set(G.nodes)
        score = -np.inf
        param = None
        i = 0

        if curr_node is None or curr_node not in G.nodes:
            curr_node = random.choice(list(candidates))

        while i<=max_iter and len(candidates) > 0:
            curr_params = G.nodes[curr_node]
            score, _ = self.evaluate_node(curr_params)
            visited_nodes.add(curr_node)
            candidates = candidates - visited_nodes
            neighbors = list(G.neighbors(curr_node))

            if not neighbors:
                l_nodes = list( set(G.nodes) - {curr_node})
                curr_node = random.choice(l_nodes)
                curr_params = G.nodes[curr_node]
                score, _ = self.evaluate_node(curr_params)

            # explore neighbours
            scores = {node:self.evaluate_node(G.nodes[node])[0] for node in neighbors}
            if score <  max(scores.values()):
                argmax_score = max(scores, key=scores.get)
                score = scores[argmax_score]
                param = G.nodes[argmax_score]

            # update current node
            i += 1
        return param, score, time.time() - start_time

    def evaluate_node(self, node_params):
        """
        Evaluate a node - dict
        """
        start_time = time.time()
        model = Pipeline(steps=[('preprocessor', self.preprocessor_step), ('classifier', self.__method__.set_params(**node_params))])
        score = cross_val_score(model, self.X, self.y, cv=self.__fold_num__, scoring=self.__metric__, n_jobs =-1).mean()
        return score, time.time() - start_time

    def build_graph_search_space(self):
        """
        Build a graph of the search space using nx
        """
        if self.G is None:
            G = nx.Graph()
            nodes = self.get_full_grid()
            node_mapping = {i: params for i, params in enumerate(nodes)}
            node_ids = list(node_mapping.keys())

            for i, node in enumerate(nodes):
                t_node = dict(zip(self.__search_spaces__.keys(), node))
                G.add_node(i, **t_node)

            # edges between nodes with hamming distance
            # hamming distance is just the number of different elements between two nodes
            # it's not optimal especially for large search spaces + doesn't take into account the actual parameter values
            for i in range(len(node_ids)):
                for j in range(i+1, len(node_ids)):
                    if sum([1 for x, y in zip(nodes[i], nodes[j]) if x != y]) == 1:
                        G.add_edge(i, j)

            self.G = G
        return self.G



    def get_neighbours(self, curr_params):
        """
        Get the neighbours of a node
        ----------
        curr_params: tuple - current parameters
        """
        return list(self.G.neighbors(curr_params))

    # utils
    def check_param_space(self):
        return set(self.__search_spaces__.keys()) <= set(self.__method__.get_params().keys()) and (set(self.__search_spaces__.keys()) > set())

    def get_full_grid(self):
        """
        returns a list of all possible combinations of the search space
        """
        return list(itertools.product(*self.__search_spaces__.values()))

