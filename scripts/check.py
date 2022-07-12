# Tabnine::sem
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearnex import patch_sklearn

from sklearn.preprocessing import (
    MaxAbsScaler,
    StandardScaler,
    add_dummy_feature,
    add_dummy_feature,
    Binarizer,
    KBinsDiscretizer,
)


data = pd.read_csv("../data/train.csv", index_col=0)
data.to_pickle("../data/train.pkl")
