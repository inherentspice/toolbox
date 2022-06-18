from toolbox.utils import assess_classifiers
import numpy as np
import pandas as pd

def test_feature_wrong_shape():
    X = np.random.randint(10, size=(40))
    y = np.random.randint(low=0, high=2, size=(40))
    test = assess_classifiers(X,y)
    assert isinstance(test, pd.DataFrame)

def test_target_invalid():
    X = np.random.randint(10, size=(40))
    y = np.random.randint(low=0, high=1, size=(40))
    test = assess_classifiers(X, y)
    assert isinstance(test, str)
