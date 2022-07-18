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
            "binary__v_1": "Binary",
            "binary__v_11": "Binary",
            "binary__v_14": "Binary",
            "binary__v_26": "Binary",
            "binary__v_27": "Binary",
            "binary__v_28": "Binary",
            "binary__v_30": "Binary",
            "binary__v_9": "Binary",
            "nominal__v_12": "Nominal",
            "nominal__v_18": "Nominal",
            "nominal__v_20": "Nominal",
            "nominal__v_21": "Nominal",
            "nominal__v_25": "Nominal",
            "nominal__v_3": "Nominal",
            "nominal__v_32": "Nominal",
            "nominal__v_4": "Nominal",
            "ordinal__v_0": "Ordinal",
            "ordinal__v_10": "Ordinal",
            "ordinal__v_13": "Ordinal",
            "ordinal__v_15": "Ordinal",
            "ordinal__v_17": "Ordinal",
            "ordinal__v_19": "Ordinal",
            "ordinal__v_22": "Ordinal",
            "ordinal__v_23": "Ordinal",
            "ordinal__v_24": "Ordinal",
            "ordinal__v_29": "Ordinal",
            "ordinal__v_31": "Ordinal",
            "ordinal__v_33": "Ordinal",
            "ordinal__v_5": "Ordinal",
            "ordinal__v_6": "Ordinal",
            "ratio__v_16": "Ratio",
            "ratio__v_2": "Ratio",
            "ratio__v_34": "Ratio",
            "ratio__v_35": "Ratio",
            "ratio__v_36": "Ratio",
            "ratio__v_37": "Ratio",
            "ratio__v_38": "Ratio",
            "ratio__v_39": "Ratio",
            "ratio__v_40": "Ratio",
            "ratio__v_7": "Ratio",
            "ratio__v_8": "Ratio",
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


# for c in ratio:
#     final_data.rename(columns={c:"ratio__"+c}, inplace=True)
#     final_data.sort_index(axis=1,inplace=True)
#     final_pred_data.rename(columns={c:"ratio__"+c}, inplace=True)
#     final_pred_data.sort_index(axis=1,inplace=True)
#     master_data.rename(columns={c:"ratio__"+c}, inplace=True)
#     master_data.sort_index(axis=1,inplace=True)
#     given_data.rename(columns={c:"ratio__"+c}, inplace=True)
#     given_data.sort_index(axis=1,inplace=True)
#     data_logit.rename(columns={c:"ratio__"+c}, inplace=True)
#     data_logit.sort_index(axis=1,inplace=True)
#     baseline_prediction_data.rename(columns={c:"ratio__"+c}, inplace=True)
#     baseline_prediction_data.sort_index(axis=1,inplace=True)
