from sklearn.svm import SVC
import time
from Annealer import Annealer

class BigDaddyWrapper:
    def __init__(self, data, feature_structure, time_limit=60, max_annealing_iterations = 100000):
        self.time_limit = time_limit * 60 # internally use seconds
        self.classifiers = []
        self.data = data    # TODO: Do we want to do train/test split here already?
        self.feature_structure = feature_structure
        self.classifiers = BigDaddyWrapper.default_classifiers()
        self.model = None
        self.max_annealing_iterations = max_annealing_iterations

    def default_classifiers():
        return [
            (SVC(), {'coef0': [0.0, 0.5, 1.0, 2.0, 2.5, 3.0],
               'degree': [2, 3, 4, 5,6],
               'gamma': ['scale', 'auto'],
               'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
               'C': [0.1, 1, 10, 100, 200]})
        ]
    # TODO: Add remaining classifiers + params
    # TODO: Add option to add / set your own.

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
            (iterations, grid_len) = self.estimate_iteration_time(classifier, params, time_chunk)  # TODO: Do we want to retrieve model from this as well (e.g. if even one iteration takes a long time)?
            
            # Then train classifier
            annealer = Annealer(classifier, self.feature_structure, params, iterations, data=self.data)
            # Do grid search if feasible witin time limit, else do SA
            (best_params, score, training_time) = annealer.grid_search() if iterations > grid_len else annealer.simulation_annealing(1, 10e-03)

            print(f"Finished {classifier}, score: {score}, time: {training_time}s")
            if score >  best_score:
                best_score = score
                best_model = (classifier, best_params)

        # TODO: Train an actual model that can be used for prediction!
        return (best_score, best_model)

    def estimate_iteration_time(self, classifier, params, time_limit):
        estimation_time_limit = time_limit / 10 # Use max 10% of time limit for estimating number of iterations
        max_iterations = 1000 # Maximal 1000 iterations for estimating number of iterations
        print("Estimate iterations")
        annealer = Annealer(classifier, self.feature_structure, params, 100, data=self.data)
        grid = annealer.get_full_grid()
        print(f"Grid len = {len(grid)}")
        limit = min(max_iterations, len(grid))
        start = time.time()
        # Use the first (random) parameter combinations
        for i in range(limit):
            annealer.evaluate_params(grid[i])
            current = time.time() - start
            print(f"i={i}, current={current}, limit={estimation_time_limit}")
            if current < estimation_time_limit and i < limit - 1:
                continue

            # Estimation time limit reached
            time_per_iteration = current / i
            iterations = min((time_limit - current) / time_per_iteration, self.max_annealing_iterations)
            print(f"Iterations: {iterations}")
            return (iterations, len(grid))

