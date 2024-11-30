import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

def find_means_between_subsequent_x(df, feature):
    # ([1,2,3][:-1] + [1,2,3][1:]) / 2
    # ([1,2] + [2,3]) / 2
    # [3,5] / 2
    # [1.5,2.5]
    means = (df[feature].values[:-1] + df[feature].values[1:]) / 2
    return means

def calculate_se_on_split(df, mean):
    sse = ((df['Target'] - mean) ** 2).sum()
    return sse

def calculate_single_sse_on_split(df):
    mean = df['Target'].mean()
    sse = calculate_se_on_split(df, mean)
    return sse

def calculate_total_sse_on_split(df, feature, value):
    # Split the DataFrame into left and right based on the mean value
    left_sse = calculate_single_sse_on_split(df[df[feature] <= value])
    right_sse = calculate_single_sse_on_split(df[df[feature] > value])

    # Return the total SSE for this split
    total_sse = left_sse + right_sse
    return total_sse

def root(df_single_feature_sorted, single_feature):

    means = find_means_between_subsequent_x(df_single_feature_sorted, single_feature)
    print(means)
    sse = []
    for mean in means:
        sse.append(calculate_total_sse_on_split(df_single_feature_sorted, single_feature, mean))
    print('sse', sse)
    # todo chose the lowest sse to split

    mse = calculate_single_sse_on_split(df_single_feature_sorted)
    print(mse)

def make_random_forest(df):
    # ToDo select multiple features
    single_feature = select_random_feature(df, 1)[0]
    print('single feature: ',single_feature)
    df_single_feature = df[[single_feature, 'Target']]
    df_single_feature_sorted = df_single_feature.sort_values(by=single_feature)
    print('df with single feature:', df_single_feature_sorted)
    root(df_single_feature_sorted, single_feature)


    print('======')

def select_random_feature(df, k):
    # remove target column from selecting a random feature.
    feature_columns = df.drop(columns=['Target']).columns
    feature = np.random.choice(feature_columns, k, replace=False)
    return feature


# steps are from 05.11. lecture. Slide 47. In the lecture there is a not a regression, but a classifier.
def random_forest(df):
    np.random.seed(RANDOM_SEED)

    datasets = bootstrap(df)
    for dataset in datasets:
        make_random_forest(dataset)

def prepare_data(df):
    # Rename 'Salary' to 'Target'
    df = df.rename(columns={'Salary': 'Target'})

    # one-to-n:
    return pd.get_dummies(df, columns=['Position'])

    # label encoding:
    # label_encoder = LabelEncoder()
    # df['Position'] = label_encoder.fit_transform(df['Position'])
    # return df

def main():
    print("Hello Dmytro, Michael and Adam!")
    df = pd.read_csv('../data/our/ass2-test-dataset-salary.csv')

    df = prepare_data(df)
    random_forest(df)


if __name__ == "__main__":
    main()
