import warnings
import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 50
from sklearn import set_config
set_config(display="diagram")
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.compose import make_column_selector

class ColumnSelectors:
    def __init__(self, default=None):
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
        self.ordinal_cols = [
            i for i in self.dtype_info if self.dtype_info[i] == "Ordinal"
        ]
        self.nominal_cols = [
            i for i in self.dtype_info if self.dtype_info[i] == "Nominal"
        ]
        self.binary_cols = [
            i for i in self.dtype_info if self.dtype_info[i] == "Binary"
        ]
        self.ratio_cols = [i for i in self.dtype_info if self.dtype_info[i] == "Ratio"]
        self.ordinal = make_column_selector(
            pattern="|".join(self.ordinal_cols),
        )
        self.nominal = make_column_selector(
            pattern="|".join(self.nominal_cols),
        )
        self.binary = make_column_selector(
            pattern="|".join(self.binary_cols),
        )
        self.ratio = make_column_selector(
            pattern="|".join(self.ratio_cols),
        )

    def ordinal_selector(self):
        return self.ordinal

    def nominal_selector(self):
        return self.nominal

    def binary_selector(self):
        return self.binary

    def ratio_selector(self):
        return self.ratio
    

def plot_mean_std_max(d_coll: list[tuple[float, float, float, float]]):
    """
    :param d_coll: list[tuple[float, float, float, float]
    list of data points in the format
    (abcissa, mean, std, max)
    """
    sns.set()
    ddx = [x for x, y, u, r in d_coll]
    ddc = [y for x, y, u, r in d_coll]
    ddep = [y + u for x, y, u, r in d_coll]
    dden = [y - u for x, y, u, r in d_coll]
    ddem = [r for x, y, u, r in d_coll]
    plt.plot(ddx, ddc, "b", label="\u00b5")
    plt.plot(ddx, ddep, "r", label="\u03c3" + "+")
    plt.plot(ddx, dden, "g", label="\u03c3" + "-")
    plt.plot(ddx, ddem, "y", label="\u03c3" + "max")
    fig = plt.fill_between(ddx, ddep, dden, alpha=0.5)
    fig = plt.legend()
    plt.show()