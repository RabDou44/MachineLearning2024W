from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import time
from Annealer import Annealer
import itertools

class BigDaddyWrapper:
    def __init__(self, data, feature_structure, time_limit=60, max_annealing_iterations = 100000):
        self.time_limit = time_limit * 60 # internally use seconds
        self.classifiers = []
        self.data = data
        self.feature_structure = feature_structure
        self.classifiers = BigDaddyWrapper.default_classifiers()
        self.model = None
        self.max_annealing_iterations = max_annealing_iterations

    def default_classifiers():
        return [
            (KNeighborsClassifier(),{
                'n_neighbors': [1,2,3,4,5,7,10,15,20,50],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10,20,30,40,50],
                'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
                'n_jobs': [-1]
            }),
            (SVC(), {
                'coef0': [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 2.5, 3.0],
               'degree': [2, 3, 4, 5,6],
               'gamma': ['scale', 'auto'],
               'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
               'C': [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
               'shrinking': [True, False],
               'probability': [True, False],
               'class_weight': [None, 'balanced'],
               'decision_function_shape': ['ovo', 'ovr']
            }),            
            (DecisionTreeClassifier(),{
                'criterion': ["gini", "entropy", "log_loss"],
                'splitter': ["best", "random"],
                'max_depth': [None, 1,2,3,4,5,7,10,15,20,50],
                'min_samples_split': [2,3,4,5,7,10,15,20],
                'min_samples_leaf': [1,2,3,4,5,7,10],
                'max_features': [None, 'sqrt', 'log2'],
                'max_leaf_nodes': [None, 10,20,30,40,50,100], 
                'min_impurity_decrease': [0.0, 0.01, 0.02, 0.03, 0.05, 0.1],
                'class_weight': [None, 'balanced']
            }),
            (RandomForestClassifier(),{
                'n_estimators': [3,5,10,20,30,50,75,100,150,200],
                'criterion': ["gini", "entropy", "log_loss"],
                'max_depth': [None, 1,2,3,4,5,7,10,15,20,50],
                'min_samples_split': [2,3,4,5,7,10,15,20],
                'min_samples_leaf': [1,2,3,4,5,7,10],
                'max_features': [None, 'sqrt', 'log2'],
                'max_leaf_nodes': [None, 10,20,30,40,50,100], 
                'min_impurity_decrease': [0.0, 0.01, 0.02, 0.03, 0.05, 0.1],
                'class_weight': [None, 'balanced']
            }),
            (MLPClassifier(),{
                'hidden_layer_sizes': BigDaddyWrapper.get_hidden_layer_sizes([10,20,50,100], 3),
                'activation': ['relu', 'tanh', 'logistic', 'identity'],
                'solver': ['adam', 'sgd', 'lbfgs'],
                'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1], 
                'batch_size': ['auto', 10, 20, 50, 100, 200],
                'learning_rate': ['constant', 'invscaling', 'adaptive'], 
                'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                'max_iter': [200, 500, 1000],
                'shuffle': [True, False], 
                'early_stopping': [True, False]
            })
        ]      

    def get_hidden_layer_sizes(sizes, depth):
        combinations = []
        for r in range(1, depth+1):
            combinations.extend(itertools.product(sizes, repeat=r))

        return combinations

    def train_model(self):
        counter = 0
        best_score = 0
        best_model = None
        start = time.time()
        for (classifier, params) in self.classifiers:
            print(f"Start with classifier {classifier}")
            # First estimate number of iterations based on time limit
            elapsed = time.time() - start
            time_chunk = (self.time_limit - elapsed) / (len(self.classifiers) - counter) # Calculate time chunk for classifier based on remaining time (and classifiers)
            (iterations, grid_len) = self.estimate_iteration_time(classifier, params, time_chunk)
            
            # Then train classifier
            annealer = Annealer(classifier, self.feature_structure, params, iterations, data=self.data)
            # Do grid search if feasible witin time limit, else do SA
            (best_params, score, training_time) = annealer.grid_search() if iterations > grid_len else annealer.simulation_annealing_fast(1, 10e-03)
            counter += 1
            print(f"Finished {classifier}, score: {score}, time: {training_time}s")
            if score >  best_score:
                best_score = score
                best_model = (classifier, best_params)

        (best_classifier, best_params) = best_model
        print(f"================ Best classifier: {best_classifier}, params: {best_params}, score: {best_score} ================")

        annealer = Annealer(best_classifier, self.feature_structure, data=self.data)
        self.model = annealer.train_model(best_params)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

    def estimate_iteration_time(self, classifier, params, time_limit):
        warmup_time = min(3, time_limit / 100)
        estimation_time_limit = time_limit / 10 # Use max 10% of time limit for estimating number of iterations
        max_iterations = 1000 # Maximal 1000 iterations for estimating number of iterations
        annealer = Annealer(classifier, self.feature_structure, params, 100, data=self.data)
        grid = annealer.get_full_grid()
        print(f"Grid len: {len(grid)}")
        limit = min(max_iterations, len(grid))

        # Warmup CPU
        start = time.time()
        for i in range(limit):
            annealer.evaluate_params(grid[i])
            if time.time() - start < warmup_time:
                break

        start = time.time()
        # Use the first (random) parameter combinations
        for i in range(limit):
            annealer.evaluate_params(grid[i])
            current = time.time() - start
            if current < estimation_time_limit and i < limit - 1:
                continue

            # Estimation time limit reached
            time_per_iteration = current / (i+1)
            iterations = min((time_limit - current - warmup_time) / time_per_iteration, self.max_annealing_iterations)
            print(f"Iterations: {iterations}")
            return (iterations, len(grid))
