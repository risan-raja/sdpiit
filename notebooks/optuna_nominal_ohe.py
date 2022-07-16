import os

import optuna

os.environ['NEPTUNE_PROJECT'] = "mlop3n/SDP"
os.environ['NEPTUNE_NOTEBOOK_PATH'] = "PycharmProjects/sdpiit/notebooks/Pipeline_components_builder.ipynb"
from sklearnex import patch_sklearn

patch_sklearn()
import pandas as pd
from sklearn import set_config
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.multiclass import *
from sklearn.neural_network import *
from sklearn.preprocessing import *
from sklearn.dummy import *

pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 50
set_config(display="diagram")
warnings.filterwarnings("ignore")

# import seaborn as sns
from joblib import parallel_backend
from joblib.memory import Memory

# sns.set()
from helpers import ColumnSelectors
from helpers import DFCollection
import gc

CACHE_DIR = Memory(location='../data/joblib_memory/')
OPTUNA_DB = "postgresql+psycopg2://postgres:302492@localhost:5433/optuna"


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
class_labels, n_classes, class_priors = class_distribution(final_data.target.to_numpy().reshape(-1, 1))

ohe_nominal_data = OneHotEncoder(sparse=False, drop="first").fit_transform(
    nominal_data.drop(['nominal__v_12', 'nominal__v_21'], axis=1))
target = final_data.target
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
sss.get_n_splits(ohe_nominal_data, final_data.target)
for train_index, test_index in sss.split(ohe_nominal_data, final_data.target):
    X_train, X_test = ohe_nominal_data[train_index], ohe_nominal_data[test_index]
    y_train, y_test = target[train_index], target[test_index]


def objective(trial: optuna.trial.Trial):
    highs = [846, 564, 423, 282, 188, 141, 94, 47, 36, 18, 12, 9]
    lows = [int(x * 0.5) for x in highs]
    h_layers = trial.suggest_int("n_layers", 1, 12)
    layer_nm = "l_"
    layers = [trial.suggest_int(layer_nm + str(idx), low_, high_) for idx, low_, high_ in
              zip(list(range(h_layers)), lows, highs)]
    tuple_layers = tuple(layers)
    # layer_1 = trial.suggest_int("n_layers", 433,900),
    # layer_sizes = [trial.suggest_int("n_layers", 433,900),]

    model_params = {
        'activation': trial.suggest_categorical("activation", ['logistic', 'tanh', 'relu']),
        'alpha': trial.suggest_float("alpha", 1e-04, 1e-03),
        'beta_1': trial.suggest_float("beta_1", 1e-03, 1),
        'beta_2': trial.suggest_float("beta_2", 1e-04, 1),
        'early_stopping': True,
        'batch_size': trial.suggest_int("batch_size", 200, 1000),
        'epsilon': 1e-08,
        'hidden_layer_sizes': tuple_layers,
        'learning_rate': 'invscaling',
        'learning_rate_init': trial.suggest_float("learning_rate_init", 1e-5, 1e-3, log=True),
        'max_iter': trial.suggest_int("n_estimators", 500, 900),
        'momentum': 0.9,
        'n_iter_no_change': 10,
        'power_t': trial.suggest_float("power_t", 0.2, 0.8, step=0.01),
        'random_state': 42,
        'shuffle': True,
        'solver': trial.suggest_categorical("solver", ['sgd', 'adam', 'lbfgs']),
        'validation_fraction': 0.2,
        'tol': 0.00001,
        'verbose': False,
        'warm_start': False
    }

    # params_model = model.get_params() 
    model = MLPClassifier(**model_params)
    m_class_methods = {"ovo": OneVsOneClassifier(model, n_jobs=2), "ovr": OneVsRestClassifier(model, n_jobs=2)}
    mclass_model = m_class_methods[trial.suggest_categorical("m_class_method", ['ovo', "ovr"])]
    trial.set_user_attr("DATASET", "One Hot Encoded Nominal Data")
    trial.set_user_attr("Model", "MLPClassifier (adam)")
    with parallel_backend('loky'):
        mclass_model.fit(X_train, y_train)
    y_pred_test = mclass_model.predict(X_test)
    y_pred_train = mclass_model.predict(X_train)
    metric1 = f1_score(y_test, y_pred_test, average='macro', labels=[0, 1, 2])
    metric2 = f1_score(y_train, y_pred_train, average='macro', labels=[0, 1, 2])
    return metric1, metric2 - metric1

    return clf.score(X_valid, y_valid)


if __name__ == '__main__':
    study = optuna.create_study(study_name="MLP_Layers=multi.test.1",
                                sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
                                storage=OPTUNA_DB,
                                directions=["maximize", "minimize"],
                                load_if_exists=True)
    with parallel_backend('loky'):
        study.optimize(objective, show_progress_bar=True, gc_after_trial=True, n_jobs=2, n_trials=500)
