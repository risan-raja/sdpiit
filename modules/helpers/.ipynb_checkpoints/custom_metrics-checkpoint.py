import pandas as pd
import numpy as np


class CustomMetrics:
    def __init__(self):
        pass

    @staticmethod
    def iv_woe(data, target, bins=10, show_woe=False):
        """
        :params data: pandas.DataFrame
        :params target: str
        :params bins: int
        :params show_woe: bool
        :returns newDF: pandas.DataFrame, woeDF: pandas.DataFrame
        """
        # Empty Dataframe
        newDF, woeDF = pd.DataFrame(), pd.DataFrame()
        # Extract Column Names
        cols = data.columns
        # Run WOE and IV on all the independent variables
        for ivars in cols[~cols.isin([target])]:
            if (data[ivars].dtype.kind in "bifc") and (
                len(np.unique(data[ivars])) > 1000
            ):
                binned_x = pd.qcut(data[ivars], bins, duplicates="drop")
                d0 = pd.DataFrame({"x": binned_x, "y": data[target]})
            else:
                d0 = pd.DataFrame({"x": data[ivars], "y": data[target]})
            d0 = d0.astype({"x": str})
            d = d0.groupby("x", as_index=False, dropna=False).agg(
                {"y": ["count", "sum"]}
            )
            d.columns = ["Cutoff", "N", "Events"]
            d["% of Events"] = np.maximum(d["Events"], 0.5) / d["Events"].sum()
            d["Non-Events"] = d["N"] - d["Events"]
            d["% of Non-Events"] = (
                np.maximum(d["Non-Events"], 0.5) / d["Non-Events"].sum()
            )
            d["WoE"] = np.log(d["% of Non-Events"] / d["% of Events"])
            d["IV"] = d["WoE"] * (d["% of Non-Events"] - d["% of Events"])
            d.insert(loc=0, column="Variable", value=ivars)
            print(
                "Information value of " + ivars + " is " + str(round(d["IV"].sum(), 6))
            )
            temp = pd.DataFrame(
                {"Variable": [ivars], "IV": [d["IV"].sum()]}, columns=["Variable", "IV"]
            )
            newDF = pd.concat([newDF, temp], axis=0)
            woeDF = pd.concat([woeDF, d], axis=0)
            # Show WOE Table
            if show_woe == True:
                print(d)
        return newDF, woeDF
