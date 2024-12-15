# RandomForestLLM
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

class TreeNodeLLM:
    def __init__(self, max_depth=None, min_leaf_size=1):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X, y, depth=0):
        # If only one unique value in the target, return it
        if len(np.unique(y)) == 1:
            self.value = np.mean(y)
            return
        
        # If max depth or min leaf size is reached, return the mean of y
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) <= self.min_leaf_size:
            self.value = np.mean(y)
            return
        
        # Find the best split using SSE
        best_sse = float('inf')
        best_left_y = None
        best_right_y = None
        best_left_X = None
        best_right_X = None
        best_feature_index = None
        best_threshold = None
        
        # Try every feature and every threshold value for splitting
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) > 0 and len(right_y) > 0:
                    sse = self._calculate_sse(left_y, right_y)
                    if sse < best_sse:
                        best_sse = sse
                        best_left_y = left_y
                        best_right_y = right_y
                        best_left_X = X[left_mask]
                        best_right_X = X[right_mask]
                        best_feature_index = feature_index
                        best_threshold = threshold
        
        # Set the best split and recursively build the tree
        self.feature_index = best_feature_index
        self.threshold = best_threshold
        self.left = TreeNodeLLM(self.max_depth, self.min_leaf_size)
        self.right = TreeNodeLLM(self.max_depth, self.min_leaf_size)
        
        # Recursively fit left and right subtrees
        self.left.fit(best_left_X, best_left_y, depth + 1)
        self.right.fit(best_right_X, best_right_y, depth + 1)
        
    def _calculate_sse(self, left_y, right_y):
        """ Calculate the Sum of Squared Errors (SSE) for a split """
        left_sse = np.sum((left_y - np.mean(left_y)) ** 2)
        right_sse = np.sum((right_y - np.mean(right_y)) ** 2)
        return left_sse + right_sse

    def predict(self, X):
        """ Predict using the decision tree """
        if self.value is not None:
            return np.full(X.shape[0], self.value)
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask
        predictions = np.zeros(X.shape[0])
        predictions[left_mask] = self.left.predict(X[left_mask])
        predictions[right_mask] = self.right.predict(X[right_mask])
        return predictions


class RandomForestLLM(BaseEstimator, RegressorMixin):
    def __init__(self, n_trees=100, max_depth=None, min_leaf_size=1, bootstrap=True):
        """
        RandomForestLLM constructor.
        :param n_trees: Number of trees in the forest (default 100)
        :param max_depth: Maximum depth of each tree (default None, i.e., no limit)
        :param min_leaf_size: Minimum number of samples in each leaf node (default 1)
        :param bootstrap: Whether to use bootstrap sampling for training trees (default True)
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        """
        Fit the random forest model.
        :param X: Training data
        :param y: Target labels
        """
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap sampling
            if self.bootstrap:
                X_resampled, y_resampled = resample(X, y)
            else:
                X_resampled, y_resampled = X, y
            
            # Create and fit a new tree
            tree = TreeNodeLLM(max_depth=self.max_depth, min_leaf_size=self.min_leaf_size)
            tree.fit(X_resampled, y_resampled)
            self.trees.append(tree)
        return self

    def predict(self, X):
        """
        Predict using the random forest model.
        :param X: Data to predict
        :return: Predicted values
        """
        tree_predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            tree_predictions[:, i] = tree.predict(X)
        
        # Return the average of predictions from all trees
        return np.mean(tree_predictions, axis=1)

    def fit_transform(self, X, y):
        """
        Fit the model and transform (predict) the training data.
        :param X: Training data
        :param y: Target labels
        :return: Predicted values for training data
        """
        self.fit(X, y)
        return self.predict(X)

    def score(self, X, y):
        """
        Calculate the R^2 score of the model.
        :param X: Data to predict
        :param y: True target values
        :return: R^2 score
        """
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

