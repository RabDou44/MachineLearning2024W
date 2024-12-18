import numpy as np

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

    def train(self, X, y):
        """
        Call this function after the Tree node creation.

        This function determines:
         the node type, if it is the leaf or tree.
         the node mean value, which is mean of target. It is non-null only by leaf.
         the best feature which provides the most fine-grade tuning.
         the split value which tells if you found the node, or you need to traverse left or right.
         the left and right nodes in case the current node is a tree.

        :param X: is the dataset we are training on
        :param y: is the target values for given X.
        """

        if self.depth >= self.max_depth or len(X) < 2:
            self.is_leaf = True
            self.mean_value = y.mean()
            return

        # Choose the best feature and split value
        best_feature = None
        best_split = None
        best_sse = float('inf')
        for col_idx in np.arange(X.shape[1]):   # Iterate all columns
            order = np.argsort(X[:,col_idx])    # Get the indices that would provide a sorting w.r.t. the current feature
            split, sse = self.find_best_split(X[order, col_idx], y[order])   # Can we do this without reordering the actual arrays? Would passing col idx be faster than filtering the col?
            if not sse == None and sse < best_sse:
                best_sse = sse
                best_feature = col_idx
                best_split = split

        if best_feature is None or best_split is None or best_sse == 0.0:
            self.is_leaf = True
            self.mean_value = y.mean()
            return

        self.feature = best_feature
        self.split_value = best_split

        # Create child nodes
        left_X = X[X[:, best_feature] <= best_split]
        left_y = y[X[:, best_feature] <= best_split]
        right_X = X[X[:, best_feature] > best_split]
        right_y = y[X[:, best_feature] > best_split]

        assert(left_y.size > 0)
        assert(right_y.size > 0)

        self.left = TreeNode(max_depth=self.max_depth, depth=self.depth + 1)
        self.right = TreeNode(max_depth=self.max_depth, depth=self.depth + 1)

        # Recursively fit child nodes
        self.left.train(left_X, left_y)
        self.right.train(right_X, right_y)

    def find_best_split(self, X, y):
        if np.all(X == X[0]):    # All elements are the same
            return None, None

        means = []
        for i in range(1, len(X)):
            (x1, x2) = (X[i-1], X[i])
            if x1 == x2: 
                continue
            means.append((x1+x2)/2)
        best_sse = float('inf')
        best_split = None
        for mean in means:
            total_sse = self.calculate_total_sse_on_split(X, y, mean)
            # find the lowest error (Sum of Squared Errors) for a given feature.
            if total_sse < best_sse:
                best_sse = total_sse
                best_split = mean

        return best_split, best_sse
    
    def calculate_total_sse_on_split(self, X, y, mean_value):
        """
        Calculate the total SSE for a dataset split into two based on a feature threshold.

        This function divides the dataset into two subsets:
        - One where the feature values are <= mean_value
        - Another where feature values are > mean_value
        It calculates the SSE for each subset and returns the sum.

        :param X: train data
        :param y: target for splitting the data
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
        left_y = y[X <= mean_value]
        right_y = y[X > mean_value]
        
        return self.calculate_se_on_split(left_y) + self.calculate_se_on_split(right_y)
    
    def calculate_se_on_split(self, y):
        """
      Calculate the Sum of Squared Errors (SSE) for a given split using the y mean value.

      :param y: target column values
      :return: float, the calculated SSE

      Example:
          Suppose we have the following data:
          y: [1, 2, 3]
          The mean value is 2:
          Calculating the SSE gives 2:
          ((1-2)^2 + (2-2)^2 + (3-2)^2) = 1 + 0 + 1 = 2
        """
        mean = y.mean()
        return ((y - mean) ** 2).sum()


    def predict(self, x):
        if self.is_leaf:
            return self.mean_value
        if x[self.feature] <= self.split_value:
            return self.left.predict(x)
        return self.right.predict(x)
