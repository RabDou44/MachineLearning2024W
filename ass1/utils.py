from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np 


def get_pipeline(feature_structure, clf = RandomForestClassifier()):

    """
    Returns a pipeline that preprocesses the data and then applies the classifier.
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

def test():
    return 1