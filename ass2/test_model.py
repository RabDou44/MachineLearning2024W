import pandas as pd
from sklearn.model_selection import train_test_split
from random_forest import RandomForest
from sklearn.utils.estimator_checks import check_estimator
from sklearn import set_config
from utils import *

def main():
    # set_config(transform_output="pandas")
    df = pd.read_csv('../data/our/ass2-test-dataset-salary.csv')
    # column_types ={
    #     "cat": ["Position"],
    #     "bin": [],
    #     "ord": ["Level"],
    #     "cont": [],
    #     "target": "Salary"
    # }

    # evaluate_models(df, column_types, [RandomForest(n_trees=3)])

    # return
    df = pd.get_dummies(df, columns=['Position'])   # basically one-hot, later done in pipeline
    # df = df.rename(columns={'Salary': 'Target'})
    x = df.drop(columns=['Salary'])
    y = df['Salary']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

    # Fitting
    forest = RandomForest(n_trees=3)
    # check_estimator(forest)
    # return
    forest.fit(train_x.copy(deep=True), train_y.copy(deep=True))

    # Predicting
    predictions = forest.predict(test_x)
    for i, res in enumerate(test_y):
        print('===')
        print(f"Sample {i + 1}: Expectation = {res}")
        print(f"Sample {i + 1}: Prediction = {predictions[i]}")

if __name__ == "__main__":
    main()
