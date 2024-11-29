import numpy as np
import pandas as pd

from ass2.random_forest_constants import *
from ass2.regression_tree import TreeNode


# references:
# https://medium.com/@byanalytixlabs/random-forest-regression-how-it-helps-in-predictive-analytics-01c31897c1d4
# https://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/ensembles/RandomForests.pdf
# https://www.researchgate.net/figure/Pseudocode-of-random-Forest-classification-23_fig4_362631001
# https://dataaspirant.com/random-forest-algorithm-machine-learing/
# https://www.geeksforgeeks.org/random-forest-regression-in-python/

# a.k.a. bagging_random_datasets
def bootstrap(df):
    datasets = []
    for _ in range(BAGGING_RANDOM_FOREST_SET_AMOUNT):
        # replace=True means an element can be chosen multiple times.
        sampled_data = df.sample(n=BAGGING_RANDOM_FOREST_SET_SIZE, replace=True, random_state=RANDOM_SEED)
        datasets.append(sampled_data)
    return datasets


def make_random_forest(df):
    """
    1) Randomly select “k” features from total “m” features. Where k << m
    2) Among the “k” features, calculate the node “d” using the best split point.
    3) Split the node into daughter nodes using the best split.
    4) Repeat 1 to 3 steps until “l” number of nodes has been reached.
    5)Build forest by repeating steps 1 to 4 for “n” number times to create “n” number of trees.
    """
    features = select_random_features(df)
    print('features:', features)
    # calculate the best split (regression):
    # according to slides (predicting numeric values p. 49),
    # I need to use the standard deviation reduction (SDR).
    # SDR is calculated across the whole dataset standard deviation.
    # Question: for the small bagging dataset, the sd might be very high. Is it correct to use sd across the whole dataset?
    # Possible answer: Step 3:
    # https://medium.com/analytics-vidhya/a-super-simple-explanation-to-regression-trees-and-random-forest-regressors-91f27957f688
def select_random_features(df):
    total_features = df.shape[1]
    k = total_features // 2
    features = np.random.choice(total_features, k, replace=False)
    return features


def build_regressions():
    return


def combine_regressions():
    return

# steps are from 05.11. lecture. Slide 47. In the lecture there is a not a regression, but a classifier.
def random_forest(df):
    np.random.seed(RANDOM_SEED)

    datasets = bootstrap(df)
    for dataset in datasets:
        make_random_forest(dataset)
    regressions = build_regressions()
    result = combine_regressions()


def main():
    print("Hello Dmytro, Michael and Adam!")
    df = pd.read_csv('../data/our/ass2-test-dataset-salary.csv')

    random_forest(df)


if __name__ == "__main__":
    main()
