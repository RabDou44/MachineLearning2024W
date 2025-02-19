import random
import numpy as np
import math
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
        self.GX = None
        self.continuous_parameters = {param for param, values in self.__search_spaces__.items() if
                                      all(isinstance(v, (int, float)) for v in values)}

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
        print("Perform grid search")
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

            # print(f"Iteration score {index+1}: {score} with params: {dict_params}")

        print(f"""Best score: {best_score} with params: {best_params}""")
        return best_params, best_score, timing
    
    def evaluate_params(self, params):
        dict_params =  dict(zip(self.__search_spaces__.keys(), params))
        model = Pipeline(steps=[('preprocessor', self.preprocessor_step), ('classifier', self.__method__.set_params(**dict_params))])
        # cross validation
        return cross_val_score(model, self.X, self.y, cv=self.__fold_num__, scoring=self.__metric__, n_jobs =-1).mean()

    def hill_climbing(self, curr_node=None):
        # build search graph
        start_time = time.time()
        G = self.GX
        if self.G is None:
            G =  self.build_search_space2()

        # initialization
        candidates = set(G.nodes)
        curr_node = random.choice(list(candidates))
        best_params = G.nodes[curr_node]
        best_score, _ = self.evaluate_node(best_params)
        time_best = start_time
        i = 0

        while i<self.__max_iter__ and len(candidates) > 0:

            neighbors = list(G.neighbors(curr_node))
            scores = {node: self.evaluate_node(G.nodes[node])[0] for node in neighbors}

            if neighbors and best_score < max(scores.values()):
                # if there's an improvement then a new one is one replaced
                curr_node = max(scores, key=scores.get)
            else:
                curr_node = random.choice(list(candidates))


            curr_params = G.nodes[curr_node]
            score, _ = self.evaluate_node(curr_params)
            candidates = candidates - set( neighbors + [curr_node])

            if score > best_score:
                best_score = score
                best_params = curr_params
                print(f"Iteration {i+1}: New maximum {score} with params: {curr_params}")
                time_best = time.time()

            # update current node
            i += 1
        return best_params, best_score, time_best - start_time

    def evaluate_node(self, node_params):
        """
        Evaluate a node - dict
        ----------------------
        parameters:
            node_params - dictionary of parameters
        return:
            score - mean score from cross_validation
            time - time in seconds
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

    def build_search_space2(self):
        """
        Build a graph where nodes are parameter instances and edges exist between nodes
        differing by one step in continuous variables or a single categorical/discrete change.
        """
        if self.GX is None:
            self.GX = nx.Graph()
            nodes = self.get_full_grid()
            node_mapping = {i: dict(zip(self.__search_spaces__.keys(), params)) for i, params in enumerate(nodes)}


            for i, node in node_mapping.items():
                self.GX.add_node(i, **node)

            for i, node_i in node_mapping.items():
                for j, node_j in node_mapping.items():
                    if i >= j:
                        continue

                    diff_count = 0
                    for param in self.__search_spaces__.keys():
                        if node_i[param] != node_j[param]:
                            if param in self.continuous_parameters:
                                values = sorted(self.__search_spaces__[param])
                                idx_i = values.index(node_i[param])
                                idx_j = values.index(node_j[param])
                                if abs(idx_i- idx_j) == 1:
                                    diff_count += 1
                                else:
                                    diff_count += 2
                            else:
                                diff_count += 1

                    if diff_count == 1:
                        self.GX.add_edge(i, j)

        return self.GX

    def simulation_annealing(self, initial_temp = 10., final_temp = 1e-03):
        print(f"Start SA with {self.__max_iter__} iterations")
        start_time = time.time()
        G = self.build_search_space2()
        alpha = self.get_alpha(self.__max_iter__, initial_temp, final_temp)   # Calculate cooling rate alpha

        candidates = list(G.nodes)
        curr_node = random.choice(candidates)
        curr_params = G.nodes[curr_node]
        best_params = curr_params
        best_score, _ = self.evaluate_node(curr_params)

        temperature = initial_temp
        i = 0

        while i < self.__max_iter__ and temperature > final_temp:
            neighbors = list(G.neighbors(curr_node))
            if not neighbors:
                break

            next_node = random.choice(neighbors)
            next_params = G.nodes[next_node]
            next_score, _ = self.evaluate_node(next_params)

            delta = next_score - best_score
            if delta > 0 or random.uniform(0, 1) < math.exp(delta / temperature):
                curr_node = next_node
                curr_params = next_params
                if next_score > best_score:
                    best_score = next_score
                    best_params = next_params

            temperature *= alpha
            i += 1

        print(f"Finished SA with best_score={best_score}")
        return best_params, best_score, time.time() - start_time
    
    
    def get_alpha(self, iterations, initial_temp, final_temp):
        return pow(final_temp / initial_temp, 1 / iterations)


    def check_param_space(self):
        return set(self.__search_spaces__.keys()) <= set(self.__method__.get_params().keys()) and (set(self.__search_spaces__.keys()) > set())

    def get_full_grid(self):
        """
        returns a list of all possible combinations of the search space
        """

        list_params = list(itertools.product(*self.__search_spaces__.values()))
        random.shuffle(list_params)
        return list_params

