# from pymongo import MongoClient
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
import neptune.new.integrations.optuna as optuna_utils
from tqdm import tqdm, trange
from xgboost import XGBClassifier, XGBRFClassifier
# from BorutaShap import BorutaShap
import xgboost as xgb
import xgboost
from sklearn.calibration import *
# from neptune.new.integrations.xgboost import NeptuneCallback as neptxgb

pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 50
set_config(display="diagram")
warnings.filterwarnings("ignore")
import pickle
from collections import defaultdict
import neptune.new as neptune
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
import joblib

# %matplotlib inline
CACHE_DIR = Memory(location="../data/joblib_memory/")
# OPTUNA_DB = "postgresql+psycopg2://postgres:302492@localhost:5433/optuna"
from REDIS_CONFIG import REDIS_URL

os.environ["NEPTUNE_PROJECT"] = "mlop3n/SDP"
CACHE_DIR = Memory(location="../data/joblib_memory/")
OPTUNA_DB = REDIS_URL
run_params = {"directions": "maximize", "n_trials": 5}
run = neptune.init(
    project="mlop3n/SDP",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MzU4OTQ1Ni02ZDMzLTRhNjAtOTFiMC04MjQ5ZDY4MjJjMjAifQ==",
    custom_run_id="XGB.5G",
    mode="offline",
)  # your credentials
# run2 = neptune.init(
#     project="mlop3n/SDP",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MzU4OTQ1Ni02ZDMzLTRhNjAtOTFiMC04MjQ5ZDY4MjJjMjAifQ==",
#     custom_run_id="XGB.5M",
#     mode="offline",
# )  # your credentials


neptune_xgb = neptxgb(run=run, log_tree=[0, 1, 2, 3])


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


# db = DFCollection()
# column_selector = ColumnSelectors()
# # classifiers = [f() for f in cls_names]
# dtype_info = column_selector.dtype_info
# ordinal = column_selector.ordinal_cols
# nominal = column_selector.nominal_cols
# binary = column_selector.binary_cols
# ratio = column_selector.ratio_cols


# final_data = db.final_data
# final_pred_data = db.final_pred_data
# baseline_prediction_data = db.baseline_prediction_data
# data_logit = db.data_logits
# prediction_data = db.prediction_data
# master_data = db.master
# given_data = db.data

# ordinal_data, nominal_data, binary_data, ratio_data = db.categorise_data()
# nominal_categories = db.nominal_categories
# ordinal_categories = db.ordinal_categories
# class_labels, n_classes, class_priors = class_distribution(
#     final_data.target.to_numpy().reshape(-1, 1)
# )
XGBOOST_OPT_TRIAL_DATA = joblib.load("../data/xgboost_optuna_trial_data/data.pkl")


def objective(trial: optuna.trial.Trial, data=XGBOOST_OPT_TRIAL_DATA):
    # X_train, X_test, y_train, y_test = XGBOOST_OPT_TRIAL_DATA
    # data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = XGBOOST_OPT_TRIAL_DATA
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    def gen_learning_rate(epoch):
        # assert type(epoch) == 'int'
        return trial.suggest_float("learning_rate", 0, 1)

    param = {
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": 3,
        # use exact for small dataset.
        "tree_method": trial.suggest_categorical(
            "tree_method", ["exact", "approx", "hist"]
        ),
        # "updater": trial.suggest_categorical("updater",['grow_colmaker', 'grow_histmaker', 'grow_local_histmaker', 'grow_quantile_histmaker']),
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "sampling_method": "uniform",
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.2, 1.0),
        "num_parallel_tree": trial.suggest_int("num_parallel_tree", 1, 10),
    }
    if param["tree_method"] != "exact":
        param["max_bin"] = trial.suggest_int("max_bin", 256, 4096)

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        # param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, "validation-mlogloss"
    )
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=999,
        evals=[(dvalid, "validation")],
        callbacks=[
            # neptune_xgb,
            pruning_callback,
            xgboost.callback.LearningRateScheduler(gen_learning_rate),
            xgboost.callback.EarlyStopping(
                rounds=5,
                min_delta=1e-5,
                save_best=True,
                maximize=False,
                data_name="validation",
                metric_name="mlogloss",
            ),
        ],
    )
    # preds = bst.predict(dvalid)
    # pred_labels = np.rint(preds)
    ypred = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
    ypred2 = bst.predict(dtrain, iteration_range=(0, bst.best_iteration + 1))
    f1_score_test = sklearn.metrics.f1_score(valid_y, ypred, average="macro")
    f1_score_train = sklearn.metrics.f1_score(train_y, ypred2, average="macro")
    # return f1_score_test, f1_score_train-f1_score_test
    run["f1_score_test"] = f1_score_test
    run["overfitting"] = f1_score_train - f1_score_test
    return f1_score_test


def main(
    params=run_params,
):
    global run
    neptune_callback = optuna_utils.NeptuneCallback(run)
    study = optuna.create_study(
        study_name="XGB.9",
        sampler=optuna.samplers.TPESampler(
            warn_independent_sampling=False,
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), 
        storage=OPTUNA_DB,
        direction=params["directions"],
        load_if_exists=True,
    )
    with parallel_backend("loky"):
        study.optimize(
            objective,
            show_progress_bar=True,
            gc_after_trial=True,
            n_jobs=-1,
            n_trials=params["n_trials"],
            callbacks=[neptune_callback],
        )

# updater_types = ['grow_colmaker', 'grow_histmaker', 'grow_local_histmaker', 'grow_quantile_histmaker','grow_gpu_hist', 'sync', 'refresh', 'prune']
if __name__ == "__main__":
    main()
    # pass