# setup + import
from utils import *
import os
import sklearn
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle



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
    classifierKneighbors = [ SVC(kernel='linear')]
    results, model = perform_holdout(X,y, classifierKneighbors[0])
    
    # save + print
    print(results) 
    pickle.dump(model, open("./porto-knn-model.pkl", 'wb'))


if __name__ == "__main__":
    main()