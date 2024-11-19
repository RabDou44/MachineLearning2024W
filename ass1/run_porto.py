# setup + import
from utils import *
import os
import sklearn
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



def main():
    # setup + import
    data = pd.read_csv('../data_processed/porto-seguero.csv')
    data.head()

    pickle_file = open('../data_processed/porto-seguero_column_types.pkl', 'rb')
    feature_structure = pickle.load(pickle_file)
    pickle_file.close()
    feature_columns = feature_structure['bin'] + feature_structure['cat'] + feature_structure['cont'] + feature_structure['ord']
    
    X = data[feature_columns]
    y = data['target']



    # # classifierKneighbors = [ KNeighborsClassifier(n_jobs=-1, n_neighbors=k) for k in range(2, 13)]
    classifierKneighbors = [ KNeighborsClassifier(n_jobs=30, n_neighbors=1)]
    resultsKNeighbors = evaluate_models(data, feature_structure, classifierKneighbors)
    print(results_to_latex(resultsKNeighbors, "Results Porto k-NN", "porto_knn"))
    # classifierDecisionTree = [ DecisionTreeClassifier(random_state=42, max_depth=depth) for depth in range(3, 15)]
    # resultsDecisionTree = evaluate_models(data, feature_structure, classifierDecisionTree)
    # print(results_to_latex(resultsDecisionTree, "Results Porto Decision Trees", "porto_dt"))


if __name__ == "__main__":
    main()