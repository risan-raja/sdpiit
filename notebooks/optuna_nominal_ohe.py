import neptune.new.integrations.optuna as optuna_utils
import neptune.new as neptune
import sklearn
from helpers import ColumnSelectors, DFCollection
from joblib.memory import Memory
from joblib import parallel_backend
import gc
from sklearn.preprocessing import *
from sklearn.neural_network import *
from sklearn.multiclass import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.dummy import *
from sklearn import set_config
import pandas as pd
from sklearnex import patch_sklearn
import os
import optuna

patch_sklearn()

pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 50
set_config(display="diagram")
warnings.filterwarnings("ignore")

os.environ["NEPTUNE_PROJECT"] = "mlop3n/SDP"
CACHE_DIR = Memory(location="../data/joblib_memory/")
OPTUNA_DB = "postgresql+psycopg2://postgres:302492@localhost:5433/optuna"

run = neptune.init(
    project="mlop3n/SDP",
    custom_run_id="MLP-1",
    mode="async",
)  # your credentials

run_params = {"direction": ["maximize", "minimize"], "n_trials": 10, "n_layers": 4}
run["parameters"] = run_params


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

ohe_nominal_data = OneHotEncoder(sparse=False, drop="first").fit_transform(
    nominal_data.drop(["nominal__v_12", "nominal__v_21"], axis=1)
)
target = final_data.target


def convert_to_dfs(X_train, X_test, y_train, y_test, encoder):
    columns = encoder.get_feature_names_out()
    X_train = pd.DataFrame(X_train, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)
    y_train = pd.DataFrame(y_train, columns=["target"])
    y_test = pd.DataFrame(y_test, columns=["target"])
    return X_train, X_test, y_train, y_test


def load_data(ohe_nominal_data, target, encoder=encoder, as_frame=True):
    X_train, X_test, y_train, y_test = train_test_split(
        ohe_nominal_data, target, stratify=target, test_size=0.2
    )
    if as_frame:
        X_train, X_test, y_train, y_test = convert_to_dfs(
            X_train, X_test, y_train, y_test, encoder=encoder
        )
    return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test  = load_data(ohe_nominal_data, target)


def objective(trial: optuna.trial.Trial):
    global ohe_nominal_data, target, run
    X_train, X_test, y_train, y_test = load_data(
        ohe_nominal_data=ohe_nominal_data, target=target
    )
    highs = [846, 564, 423, 282, 188, 141, 94, 47, 36, 18, 12, 9]
    lows = [int(x * 0.5) for x in highs]
    # h_layers = trial.suggest_int("n_layers", 1, 12)
    h_layers = 4
    layer_nm = "l_"
    layers = [
        trial.suggest_int(layer_nm + str(idx), low_, high_)
        for idx, low_, high_ in zip(list(range(h_layers)), lows, highs)
    ]
    tuple_layers = tuple(layers)

    model_params = {
        "activation": trial.suggest_categorical("activation", ["tanh", "relu"]),
        "alpha": trial.suggest_float("alpha", 1e-04, 1e-03),
        "beta_1": trial.suggest_float("beta_1", 0.77, 1),
        "beta_2": trial.suggest_float("beta_2", 0.5, 1),
        "early_stopping": True,
        "batch_size": trial.suggest_int("batch_size", 700, 1000),
        "epsilon": 1e-08,
        "hidden_layer_sizes": tuple_layers,
        "learning_rate": "invscaling",
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 1e-5, 1e-3, log=True
        ),
        "max_iter": trial.suggest_int("epochs", 50, 900),
        "momentum": 0.9,
        "n_iter_no_change": 10,
        "power_t": trial.suggest_float("power_t", 0.2, 0.8, step=0.01),
        "random_state": 42,
        "shuffle": True,
        "solver": "adam",
        "validation_fraction": 0.1,
        "tol": 0.00001,
        "verbose": False,
        "warm_start": False,
    }

    # params_model = model.get_params()
    model = MLPClassifier(**model_params)
    m_class_methods = {
        "ovo": OneVsOneClassifier(model, n_jobs=-1),
        "ovr": OneVsRestClassifier(model, n_jobs=-1),
    }
    mclass_model = m_class_methods[
        trial.suggest_categorical("m_class_method", ["ovo", "ovr"])
    ]  # type: ignore  # type: ignore
    trial.set_user_attr("DATASET", "One Hot Encoded Nominal Data")
    trial.set_user_attr("Model", "MLPClassifier (adam)")
    with parallel_backend("loky"):
        mclass_model.fit(X_train, y_train)
    y_pred_test = mclass_model.predict(X_test)
    y_pred_train = mclass_model.predict(X_train)
    metric1 = f1_score(y_test, y_pred_test, average="macro", labels=[0, 1, 2])
    metric2 = f1_score(y_train, y_pred_train, average="macro", labels=[0, 1, 2])

    def log_model_params_by_trial(
        trial: optuna.trial.Trial, mclass_model: sklearn.multiclass.OneVsOneClassifier
    ):
        # run["trial_id"]
        models = mclass_model.estimators_
        trial_info = {
            "trial_id": trial.number,
            "model_name": models[0].__class__.__name__,
        }
        for k, v in models[0].get_params().items():
            trial_info["mlp__" + k] = v
        return trial_info

    trial_params = log_model_params_by_trial(trial=trial, mclass_model=mclass_model)
    run["trial_params"] = trial_params
    run["test__f1_macro"] = metric1

    return metric1, metric2 - metric1


def main(
    params=run_params,
):
    global run
    neptune_callback = optuna_utils.NeptuneCallback(run)
    study = optuna.create_study(
        study_name="MLP_Layers=multi.test.2",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        storage=OPTUNA_DB,
        directions=params["direction"],
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
