import numpy as np
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

    def fit_grid_search(self):
        """
        Perform grid search on the given method
        """
        # currently typical grid search
        # assert(self.check_param_space(), "The search space is not a subset of the method's parameters")
        # assert(self.__feature_structure__,"Feature structure is not defined"
        param_grid = self.get_full_grid()
        best_params = None
        best_score = - np.inf
        timing = 0

        for index, params in param_grid.iterrows():

            if index >= self.__max_iter__:
                break

            params =  dict(zip(param_grid.keys(), params))
            model = Pipeline(steps=[('preprocessor', self.preprocessor_step), ('classifier', self.__method__.set_params(**params))])

            # cross validation
            start_time = time.time()
            score = cross_val_score(model, self.X, self.y, cv=self.__fold_num__, scoring=self.__metric__, n_jobs =-1).mean()
            timing += time.time() - start_time

            if score > best_score:
                best_score = score
                best_params = params

            print(f"Iteration score {index+1}: {score} with params: {params}")

        print(f"""Best score: {best_score} with params: {best_params}""")
        return best_params, best_score, timing

    # utils
    def check_param_space(self):
        return set(self.__search_spaces__.keys()) <= set(self.__method__.get_params().keys()) and (set(self.__search_spaces__.keys()) > set())

    def get_full_grid(self):
        return pd.DataFrame(list(itertools.product(*self.__search_spaces__.values())), columns=self.__search_spaces__.keys())

