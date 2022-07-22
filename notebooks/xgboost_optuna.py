from pymongo import MongoClient
import optuna
import os

os.environ["NEPTUNE_PROJECT"] = "mlop3n/SDP"
os.environ[
    "NEPTUNE_NOTEBOOK_PATH"
] = "PycharmProjects/sdpiit/notebooks/Pipeline_components_builder.ipynb"
import warnings
from sklearnex import patch_sklearn

patch_sklearn()
import numpy as np
import pandas as pd
from category_encoders import (
    BackwardDifferenceEncoder,
    BaseNEncoder,
    BinaryEncoder,
    CatBoostEncoder,
    CountEncoder,
    GLMMEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    QuantileEncoder,
    SummaryEncoder,
    TargetEncoder,
    WOEEncoder,
)
from sklearn import set_config
from sklearn.base import clone as model_clone
from sklearn.cluster import *
from sklearn.compose import *
from sklearn.cross_decomposition import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.gaussian_process import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.multioutput import *
from sklearn.multiclass import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.utils import *
from sklearn.dummy import *
from sklearn.semi_supervised import *
from sklearn.discriminant_analysis import *
import sklearnex, daal4py

from tqdm import tqdm, trange
from xgboost import XGBClassifier, XGBRFClassifier
from BorutaShap import BorutaShap

from sklearn.calibration import *

pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 50
set_config(display="diagram")
warnings.filterwarnings("ignore")
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from joblib import parallel_backend
from joblib.memory import Memory

sns.set()
from pprint import pprint
from helpers import PolynomialWrapper as PWrapper
from helpers import NestedCVWrapper as NCVWrapper
from helpers import ColumnSelectors
import sklearn

from helpers import DFCollection
from helpers import plot_mean_std_max
from helpers import CustomMetrics
import gc

%matplotlib inline
CACHE_DIR = Memory(location="../data/joblib_memory/")
# OPTUNA_DB = "postgresql+psycopg2://postgres:302492@localhost:5433/optuna"
from REDIS_CONFIG import REDIS_URL
os.environ["NEPTUNE_PROJECT"] = "mlop3n/SDP"
CACHE_DIR = Memory(location="../data/joblib_memory/")
OPTUN_DB = REDIS_URL
run_params = {"directions": ["maximize","minimize"], "n_trials": 10}
run = neptune.init(
    project="mlop3n/SDP",
    custom_run_id="XGBRF",
    mode="async",
)  # your credentials


def allow_stopping(func):
    def wrapper():
        try:
            value = func()
            return value
            # gc.collect()
        except KeyboardInterrupt as e:
            print("Program Stopped")
        gc.collect()

    return wrapper

db = DFCollection()
column_selector = ColumnSelectors()
# classifiers = [f() for f in cls_names]
dtype_info = column_selector.dtype_info
ordinal = column_selector.ordinal_cols
nominal = column_selector.nominal_cols
binary = column_selector.binary_cols
ratio = column_selector.ratio_cols


final_data = db.final_data
final_pred_data = db.final_pred_data
baseline_prediction_data = db.baseline_prediction_data
data_logit = db.data_logits
prediction_data = db.prediction_data
master_data = db.master
given_data = db.data

ordinal_data, nominal_data, binary_data, ratio_data = db.categorise_data()
nominal_categories = db.nominal_categories
ordinal_categories = db.ordinal_categories
class_labels, n_classes, class_priors = class_distribution(
    final_data.target.to_numpy().reshape(-1, 1)
)

def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)



def main(
    params=run_params,
):
    global run
    neptune_callback = optuna_utils.NeptuneCallback(run)
    study = optuna.create_study(
        study_name="XGBRF",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        storage=OPTUNA_DB,
        directions=params["directions"],
        load_if_exists=True,
    )
    with parallel_backend("threading"):
        study.optimize(
            objective,
            show_progress_bar=True,
            gc_after_trial=True,
            n_jobs=1,
            n_trials=params["n_trials"],
            callbacks=[neptune_callback],
        )

if __name__ == "__main__":
    main()