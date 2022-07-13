from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.pipeline import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.cluster import *
from sklearn.compose import *
from sklearn.cross_decomposition import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.feature_selection import *
from sklearn.gaussian_process import *
from sklearn.model_selection import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.multioutput import *
from sklearn.naive_bayes import *
from sklearn.base import clone as model_clone
from sklearn.utils import *
from sklearn import set_config
import warnings
import pandas as pd
import numpy as np
from category_encoders import (BackwardDifferenceEncoder, BinaryEncoder, CatBoostEncoder, GLMMEncoder, MEstimateEncoder,
                               QuantileEncoder, JamesSteinEncoder, HelmertEncoder, LeaveOneOutEncoder, TargetEncoder,
                               SummaryEncoder, WOEEncoder, BaseNEncoder, CountEncoder)
from xgboost import XGBRFClassifier, XGBClassifier
from tqdm import tqdm, trange

pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 50
set_config(display="diagram")
warnings.filterwarnings("ignore")
from joblib.memory import Memory
from joblib import parallel_backend
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
sns.set()
from pprint import pprint


class ColumnSelectors:
    def __int__(self):
        self.dtype_info = {
            "v_1": "Binary",
            "v_26": "Binary",
            "v_11": "Binary",
            "v_14": "Binary",
            "v_30": "Binary",
            "v_28": "Binary",
            "v_9": "Binary",
            "v_27": "Binary",
            "v_32": "Nominal",
            "v_4": "Nominal",
            "v_3": "Nominal",
            "v_20": "Nominal",
            "v_21": "Nominal",
            "v_18": "Nominal",
            "v_25": "Nominal",
            "v_12": "Nominal",
            "v_31": "Ordinal",
            "v_15": "Ordinal",
            "v_19": "Ordinal",
            "v_13": "Ordinal",
            "v_33": "Ordinal",
            "v_17": "Ordinal",
            "v_29": "Ordinal",
            "v_23": "Ordinal",
            "v_6": "Ordinal",
            "v_24": "Ordinal",
            "v_10": "Ordinal",
            "v_5": "Ordinal",
            "v_22": "Ordinal",
            "v_0": "Ordinal",
            "v_16": "Ratio",
            "v_2": "Ratio",
            "v_8": "Ratio",
            "v_7": "Ratio",
            "v_39": "Ratio",
            "v_37": "Ratio",
            "v_38": "Ratio",
            "v_34": "Ratio",
            "v_40": "Ratio",
            "v_36": "Ratio",
            "v_35": "Ratio",
        }
        ordinal = [i for i in self.dtype_info if self.dtype_info[i] == "Ordinal"]
        nominal = [i for i in self.dtype_info if self.dtype_info[i] == "Nominal"]
        binary = [i for i in self.dtype_info if self.dtype_info[i] == "Binary"]
        ratio = [i for i in self.dtype_info if dtype_info[i] == "Ratio"]
