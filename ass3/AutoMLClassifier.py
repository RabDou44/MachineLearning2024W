#%%
import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier


class AutoMLClassifier:
    """
    | Feature/Aspect             | H2O                                                              | TPOT                                      |
    |-----------------------------|-----------------------------------------------------------------|-------------------------------------------|
    | **Purpose**                | Distributed scalable ML with AutoML                              | Automating pipeline generation for small/medium datasets |
    | **Scalability**            | Distributed, handles big data across clusters                    | Limited to single-machine (single/multi-core) systems    |
    | **Core Technology**        | Distributed in-memory ML with AutoML clusters                    | Genetic algorithms to optimize sklearn ML pipelines      |
    | **Platform**               | Runs on clusters, servers, or single machines                    | Operates on local systems (non-distributed)             |
    | **Preprocessing**          | Built-in (e.g., one-hot encoding, scaling, handling missing data)| Part of the pipeline, automatically optimized         |
    | **Algorithms Supported**   | Supports Random Forest, GBM, Deep Learning, etc.                 | Relies on scikit-learn with algorithms like SVM, Random Forest |
    | **Output**                 | Trained model (MOJO files, leaderboard, etc.)                    | Sklearn-compatible Python pipeline                     |
    | **AutoML Capabilities**    | Fully automated: model selection, tuning, ensembles              | Focused mainly on pipeline optimization               |
    | **Ease of Use**            | Powerful, requires resource and process learning                 | Simple UI but less suited for big data                 |
    """
    def __init__(self, max_time_tpot=60, max_time_h2o=60):
        self.max_time_tpot = max_time_tpot
        self.max_time_h2o = max_time_h2o
        self.tpot_model = None
        self.h2o_model = None
        self.ensemble_model = None

    def train_tpot(self, X_train, y_train):
        """
        TPOT: has the following principles:
         - **Mutation:** Altering parts of the pipeline (e.g., changing hyperparameters or replacing a model).
         - **Crossover:** Combining elements of two different pipelines to create a new pipeline.
         - **Selection:** Choosing the top-performing pipelines for reproduction and modification in the next generation.
         - **Fitness Evaluation:** Measuring pipeline performance using a scoring metric (e.g., accuracy, F1 score).

         TPOT classifier:
         - **`generations` **: Number of iterations (evolution cycles) TPOT should perform.
         - **`population_size` **: Number of pipelines in each generation (higher values cover more pipelines but take longer).
         - **`scoring` **: Metric used to evaluate the pipelines, such as `accuracy`, `roc_auc`, `f1`, etc.
         - **`verbosity` **: Controls the logging output (e.g., progress updates) during training.
         - **`max_time_mins` **: Time limit for searching for the best pipeline.

        """
        print("[*] Training TPOT Classifier")
        self.tpot_model = TPOTClassifier(generations=5, population_size=50, cv=5,
                                         scoring='accuracy',
                                         verbosity=0,
                                         random_state=42,
                                         max_time_mins=self.max_time_tpot)
        self.tpot_model.fit(X_train, y_train)
        print("[+] Finished training TPOT Classifier")

    def train_h2o(self, X_train, y_train):
        """
        - **Scalability**: H2O is designed for large datasets. It splits data across computers when running on clusters or multi-core CPUs. This makes it faster and efficient for big data use cases.
        - **Distributed Machine Learning**: Algorithms (e.g., Random Forest, GBM, XGBoost, GLM, Deep Learning) are parallelized and designed for handling rows distributed across machines.
        """
        print("[*] Training H2O AutoML...")
        h2o.init(verbose=False)
        # Convert to H2OFrame
        h2o_train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
        target = y_train.name

        # Ensure other columns are predictors
        predictors = X_train.columns.tolist()

        self.h2o_model = H2OAutoML(max_runtime_secs=self.max_time_h2o * 60,
                                   seed=42, balance_classes=True)
        self.h2o_model.train(x=predictors, y=target, training_frame=h2o_train)
        print("[+] Finished training H2O AutoML")

    def predict_tpot(self, X_test):
        if self.tpot_model is None:
            raise Exception("[-] TPOT model has not been trained yet!")
        print("[*] Making predictions with TPOT")
        return self.tpot_model.predict(X_test)

    def predict_h2o(self, X_test):
        if self.h2o_model is None:
            raise Exception("[-] H2O model has not been trained yet!")
        print("[*] Making predictions with H2O AutoML")
        h2o_frame = h2o.H2OFrame(X_test)
        predictions = self.h2o_model.leader.predict(h2o_frame).as_data_frame()
        return predictions['predict'].values

    def evaluate(self, y_true, y_pred):
        """
        Evaluates the final predictions using accuracy score.

        Parameters:
        - y_true: True labels
        - y_pred: Predicted labels

        Returns:
        - Accuracy score
        """
        return accuracy_score(y_true, y_pred)
