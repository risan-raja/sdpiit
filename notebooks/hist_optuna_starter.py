import optuna
from REDIS_CONFIG import REDIS_URL
import warnings
warnings.filterwarnings('ignore')
OPTUNA_DB = REDIS_URL

X = raw_data.loc[:,nominal+ordinal]
y = raw_data.target
X_train, X_test, y_train, y_test = gen_train_test(X, y, test_size=0.3)

# @allow_stopping
def objective(trial: optuna.trial.Trial):
    categories = [True] * 22
    e_stop = {"TRUE": True, "FALSE": False}
    hist_params = {
        "categorical_features": categories,
        "early_stopping": e_stop[
            trial.suggest_categorical("early_stopping", ["TRUE", "FALSE"])
        ],
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 0.999999),
        "learning_rate": trial.suggest_float("learning_rate", 0.00001, 0.999999),
        "loss": "log_loss",
        "max_bins": trial.suggest_int("max_bins", 89, 255),
        "max_depth": trial.suggest_int("max_depth", 1, 40),
        "max_iter": trial.suggest_int("max_iter", 20, 200),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 400),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 400),
        "monotonic_cst": None,
        "n_iter_no_change": 5,
        "random_state": 10,
        "scoring": "f1_macro",
        "tol": 1e-07,
        "validation_fraction": trial.suggest_float("validation_fraction", 0.1, 0.3),
        "verbose": 0,
        "warm_start": False,
    }
    clf = HistGradientBoostingClassifier(**hist_params)
    wf = OneVsRestClassifier(clf, n_jobs=-1)
    with parallel_backend("loky"):
        y_pred = wf.fit(X_train, y_train).predict(X_test)
        score = sklearn.metrics.f1_score(y_test, y_pred, average="macro")
    return score

__make_study__= 1
# objective()
if __make_study__ ==1:
    study = optuna.create_study(
        study_name="Hist.Alpha.Kaggle.5",
        sampler=optuna.samplers.TPESampler(
            warn_independent_sampling=False,seed=29,n_ei_candidates=50, n_startup_trials=50
        ),
        storage=OPTUNA_DB,
        direction="maximize",
        load_if_exists=True,
    )
    with parallel_backend("threading"):
        study.optimize(
            objective,
            gc_after_trial=True,
            n_jobs=2,
            n_trials=100,
        )
