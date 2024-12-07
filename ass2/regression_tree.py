from random_forest_constants import *

class TreeNode:
    def __init__(self, max_depth, depth=0):
        self.left = None
        self.right = None
        self.feature = None
        self.split_value = None
        self.is_leaf = False
        self.mean_value = None
        self.max_depth = max_depth
        self.depth = depth

    def calculate_se_on_split(self, y, mean):
        """
      Calculate the Sum of Squared Errors (SSE) for a given split using the provided mean value.

      :param df: pandas DataFrame containing a 'Target' column
      :param mean: float, the mean value of the target variable
      :return: float, the calculated SSE

      Example:
          Suppose we have the following data:
          df = pd.DataFrame({'Target': [1, 2, 3]})
          The mean value is 2:
          Calculating the SSE gives 2:
          ((1-2)^2 + (2-2)^2 + (3-2)^2) = 1 + 0 + 1 = 2
        """
        return ((y - mean) ** 2).sum()

    def calculate_single_sse_on_split(self, y):
        mean = y.mean()
        return self.calculate_se_on_split(y, mean)

    def calculate_total_sse_on_split(self, X, y, mean_value):
        """
        Calculate the total SSE for a dataset split into two based on a feature threshold.

        This function divides the dataset into two subsets:
        - One where the feature values are <= mean_value
        - Another where feature values are > mean_value
        It calculates the SSE for each subset and returns the sum.

        :param df: pandas DataFrame containing the data
        :param feature: str, column name used for splitting the data
        :param mean_value: float, the mean value to split the data on
        :return: float, the total SSE of both splits

        Example:
            Suppose the DataFrame:
            data = pd.DataFrame({'Feature': [1, 2, 3, 4], 'Target': [2, 3, 4, 5]})
            We split it with:
            feature = 'Feature'
            mean_value = 2.5
            The data splits into:
            Left subset (Feature <= 2.5): Target = [2, 3]
            Right subset (Feature > 2.5): Target = [4, 5]
            SSE calculation:
             # Left SSE: ((2-2.5)^2 + (3-2.5)^2) = 0.5
             # Right SSE: ((4-4.5)^2 + (5-4.5)^2) = 0.5
             # Total SSE = 0.5 + 0.5 = 1.0
        """
        left_sse = self.calculate_single_sse_on_split(y[X <= mean_value])
        right_sse = self.calculate_single_sse_on_split(y[X > mean_value])
        return left_sse + right_sse

    def find_best_split(self, X, y):
        # ([1,2,3][:-1] + [1,2,3][1:]) / 2
        # ([1,2] + [2,3]) / 2
        # [3,5] / 2
        # [1.5,2.5]
        means = (X.values[:-1] + X.values[1:]) / 2
        best_sse = float('inf')
        best_split = None
        for mean in means:
            total_sse = float(self.calculate_total_sse_on_split(X, y, mean))
            # find the lowest error (Sum of Squared Errors) for a given feature.
            if total_sse < best_sse:
                best_sse = total_sse
                best_split = mean

        return best_split, best_sse

    def train(self, X, y):
        """
        Call this function after the Tree node creation.

        This function determines:
         the node type, if it is the leaf or tree.
         the node mean value, which is mean of target. It is non-null only by leaf.
         the best feature which provides the most fine-grade tuning.
         the split value which tells if you found the node, or you need to traverse left or right.
         the left and right nodes in case the current node is a tree.

        :param df: is the dataset we are training on
        """

        if self.depth >= self.max_depth or len(X) < 2:
            self.is_leaf = True
            self.mean_value = y.mean()
            return

        # Choose the best feature and split value
        best_feature = None
        best_split = None
        best_sse = float('inf')

        for feature in X.columns:
            sorted_X = X.sort_values(by=feature)
            sorted_y = y.reindex(sorted_X.index)
            split, sse = self.find_best_split(sorted_X[feature], sorted_y)
            if sse < best_sse:
                best_sse = sse
                best_feature = feature
                best_split = split

        if best_feature is None or best_split is None:
            self.is_leaf = True
            self.mean_value = y.mean()

        self.feature = best_feature
        self.split_value = best_split
        
        # Create child nodes
        left_X = X[X[best_feature] <= best_split]
        right_X = X[X[best_feature] > best_split]

        self.left = TreeNode(max_depth=self.max_depth, depth=self.depth + 1)
        self.right = TreeNode(max_depth=self.max_depth, depth=self.depth + 1)

        # Recursively fit child nodes
        self.left.train(left_X, y.reindex(left_X.index))
        self.right.train(right_X, y.reindex(right_X.index))

    def predict(self, x):
        if self.is_leaf:
            return self.mean_value
        if x[self.feature] <= self.split_value:
            return self.left.predict(x)
        return self.right.predict(x)
