import pandas as pd
from ass2.regression_tree import *


# references:
# https://medium.com/@byanalytixlabs/random-forest-regression-how-it-helps-in-predictive-analytics-01c31897c1d4
# https://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/ensembles/RandomForests.pdf
# https://www.researchgate.net/figure/Pseudocode-of-random-Forest-classification-23_fig4_362631001
# https://dataaspirant.com/random-forest-algorithm-machine-learing/
# https://www.geeksforgeeks.org/random-forest-regression-in-python/
# https://medium.com/analytics-vidhya/a-super-simple-explanation-to-regression-trees-and-random-forest-regressors-91f27957f688

def test_forest(forest):
    test_samples = [
        {
            "Level": 1,
            "Position_Business Analyst": True,
            "Position_C-level": False,
            "Position_CEO": False,
            "Position_Country Manager": False,
            "Position_Junior Consultant": False,
            "Position_Manager": False,
            "Position_Partner": False,
            "Position_Region Manager": False,
            "Position_Senior Consultant": False,
            "Position_Senior Partner": False,
        },
        {
            "Level": 4,
            "Position_Business Analyst": False,
            "Position_C-level": False,
            "Position_CEO": False,
            "Position_Country Manager": False,
            "Position_Junior Consultant": False,
            "Position_Manager": True,
            "Position_Partner": False,
            "Position_Region Manager": False,
            "Position_Senior Consultant": False,
            "Position_Senior Partner": False,
        },
        {
            "Level": 10,
            "Position_Business Analyst": False,
            "Position_C-level": False,
            "Position_CEO": True,
            "Position_Country Manager": False,
            "Position_Junior Consultant": False,
            "Position_Manager": False,
            "Position_Partner": False,
            "Position_Region Manager": False,
            "Position_Senior Consultant": False,
            "Position_Senior Partner": False,
        },
    ]
    test_df = pd.DataFrame(test_samples)
    expectation = [45000, 80000, 1000000]
    for i, sample in test_df.iterrows():
        prediction = forest.predict(sample)
        print('===')
        print(f"Sample {i + 1}: Expectation = {expectation[i]}")
        print(f"Sample {i + 1}: Prediction = {prediction}")

def random_forest(df):
    forest = RandomForest(n_trees=BAGGING_RANDOM_FOREST_SET_AMOUNT)
    forest.fit(df)
    return forest

def prepare_data(df):
    # Rename 'Salary' to 'Target'
    df = df.rename(columns={'Salary': 'Target'})

    # one-to-n:
    df = pd.get_dummies(df, columns=['Position'])

    # label encoding:
    # label_encoder = LabelEncoder()
    # df['Position'] = label_encoder.fit_transform(df['Position'])

    df.to_csv('../data/our/ass2-test-dataset-salary-prepared.csv', index=False)
    return df

def main():
    print("Hello Dmytro, Michael and Adam!")
    df = pd.read_csv('../data/our/ass2-test-dataset-salary.csv')

    df = prepare_data(df)
    forest = random_forest(df)

    test_forest(forest)

if __name__ == "__main__":
    main()
