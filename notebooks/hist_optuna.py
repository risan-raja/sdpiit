import optuna
import pandas as pd
import numpy as np
from joblib import parallel_backend
import pickle
import joblib
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
import sklearn
from REDIS_CONFIG import REDIS_URL

OPTUNA_DB = REDIS_URL





def train_test(X, y, test_size):
    """
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=10, test_size=test_size, stratify=y
    )
    return X_train, X_test, y_train, y_test

final_data_ohe = pd.read_parquet('../data/final_data_ohe')


target = final_data_ohe.target
X_train, X_test, y_train, y_test = train_test(final_data_ohe.drop(['target'],axis=1), target, test_size=0.2)

def objective(trial:optuna.trial.Trial):
    categories = [True]*130
    e_stop = {'TRUE': True, "FALSE":False}
    hist_params = {'categorical_features': categories,
                 'early_stopping': e_stop[trial.suggest_categorical("early_stopping",["TRUE","FALSE"])],
                     'l2_regularization': trial.suggest_float("l2_regularization", 0.0, 0.999999),
                     'learning_rate': trial.suggest_float("learning_rate", 0.00001, 0.999999),
                     'loss': 'log_loss',
                     'max_bins': 255,
                     'max_depth': trial.suggest_int('max_depth',1,40),
                     'max_iter': trial.suggest_int('max_iter',1,500),
                     'max_leaf_nodes': trial.suggest_int('max_leaf_nodes',5,400),
                     'min_samples_leaf': trial.suggest_int('min_samples_leaf',2,400),
                     'monotonic_cst': None,
                     'n_iter_no_change': 10,
                     'random_state': 10,
                     'scoring': 'f1_macro',
                     'tol': 1e-07,
                     'validation_fraction': trial.suggest_float("validation_fraction", 0.1, 0.3),
                     'verbose': 0,
                     'warm_start': False}
    c_select = make_column_selector(pattern='one_hot_enc*|ordinal*')
    clf = HistGradientBoostingClassifier(**hist_params)
    ct = make_column_transformer((('passthrough',c_select)),sparse_threshold=0)
    wf = make_pipeline(ct,clf)
    with parallel_backend('loky'):
        y_pred = wf.fit(X_train,y_train).predict(X_test)
        score = sklearn.metrics.f1_score(y_test,y_pred,average='macro')
    return score


# objective()
study = optuna.create_study(
        study_name="Hist.Beta",
        sampler=optuna.samplers.TPESampler(
            warn_independent_sampling=False,
        ),
        storage=OPTUNA_DB,
        direction="maximize",
        load_if_exists=True,
    )
# with parallel_backend("threading"):
study.optimize(
    objective,
    show_progress_bar=True,
    gc_after_trial=True,
    # n_jobs=2,
    n_trials=50,
)

