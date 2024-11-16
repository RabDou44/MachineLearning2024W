from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np 
import copy
import time 

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_pipeline(feature_structure, clf = RandomForestClassifier()):

    """
    Returns a pipeline that preprocesses the data and then applies the classifier.
    Parameters
    -----------
    feature_structure: dict
        A dictionary containing the feature structure of the data. The keys are 'bin', 'cat', 'cont', 'ord', and 'target'.
    clf: sklearn classifier
        The classifier to use in the pipeline.
    TODO: write  check for feature_structure` and `clf` types
    """
    categorical_preprocessor = Pipeline(
        steps=[
        ('onehot', OneHotEncoder())
        ])

    numerical_preprocessor = Pipeline([
        ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ('scaler', StandardScaler())
    ])  

    preprocessor = ColumnTransformer([
        ('categorical', categorical_preprocessor, feature_structure['cat']),
        ('numerical', numerical_preprocessor, feature_structure['cont'])
    ])

    return make_pipeline(preprocessor, clf)


def evaluate_models(data, feature_structure, classifiers):

    """
    Builds a model using the pipeline and returns it
    with split data and specified parameters 

    Parameters
    -----------
    data: pandas DataFrame
        The data to use for training and testing the model.
    feature_structure: dict
        A dictionary containing the feature structure of the data. The keys are 'bin', 'cat', 'cont', 'ord', and 'target'.
    classifiers: list of sklearn to eveluate (with set parameters)
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """

    feature_columns = feature_structure['bin'] + feature_structure['cat'] + feature_structure['cont'] + feature_structure['ord']
    X = data[feature_columns]
    y = data[feature_structure['target']]
    
    results = {}

    for clf in classifiers:
        model_holdout = get_pipeline(feature_structure, clf)
        model_cv = copy.deepcopy( model_holdout)

        # Holdout
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
        start_holdout= time.time()
        model_holdout.fit( trainX, trainY)
        end_holdout = time.time()   
        pred_y  = model_holdout.predict(testX)

        results[str(clf) + "_holdout"] =    [accuracy_score(testY, pred_y),  
                                             precision_score(testY, pred_y, average="weighted"), 
                                            recall_score(testY, pred_y,average="weighted"), 
                                            f1_score(testY, pred_y, average="weighted")]

        # Cross-validation
        cv_results = cross_validate(model_cv, X, y, cv=5, scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"], verbose=1)
        results[str(clf) + "_cv"] = cv_results
    
    return results


def perform_holdout(X, y, clf, random_state=42):
    """
    Builds a model using the pipeline and returns it
    with split data and specified parameters 

    Parameters
    -----------
    X: pandas DataFrame
        The data to use for training and testing the model.
    y: pandas Series
        The target variable.
    clf: sklearn classifier
        The classifier to use in the pipeline.
    random_state: int
        The random state to use for the train-test split.

    
    Keyword arguments:
    argument -- description
    Return: return a tuple of performance metrics and the model
    """

    model_holdout = clf
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=random_state)
    start_time = time.time()
    model_holdout.fit( trainX, trainY)
    end_time = time.time()
    pred_y  = model_holdout.predict(testX)

    results = [accuracy_score(testY, pred_y),  
             precision_score(testY, pred_y, average="weighted"), 
            recall_score(testY, pred_y,average="weighted"), 
            f1_score(testY, pred_y, average="weighted"), end_time - start_time] 

    return (results, model_holdout)

def perform_cv(X, y, clf):
    """
    Builds a model using the pipeline and returns it
    with split data and specified parameters 

    Parameters
    -----------
    X: pandas DataFrame
        The data to use for training and testing the model.
    y: pandas Series
        The target variable.
    clf: sklearn classifier
        The classifier to use in the pipeline.
    
    Keyword arguments:
    argument -- description
    Return: return a tuple of performance metrics and the model
    """

    model_cv = clf
    cv_results = cross_validate(model_cv, X, y, cv=5, scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"], verbose=1)
    
    return cv_results