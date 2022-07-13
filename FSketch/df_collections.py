import pandas as pd
from .components import ColumnSelectors


class DFCollection:
    """
    Contains all the data used.
    Upon Init all data gets loaded.
    Save method is also provided.
    """

    def __init__(self):
        self.file_path = "/home/mlop3n/PycharmProjects/sdpiit/data/"
        self.data = pd.read_parquet(
            self.file_path + "train.parquet", engine="fastparquet"
        )
        self.prediction_data = pd.read_parquet(
            self.file_path + "test.parquet", engine="fastparquet"
        )
        self.data_logits = pd.read_parquet(
            self.file_path + "data_with_ridit.hdfs", engine="fastparquet"
        )
        self.final_data = pd.read_parquet(
            self.file_path + "final_data.parquet", engine="fastparquet"
        )
        self.final_pred_data = pd.read_parquet(
            self.file_path + "final_pred_data.parquet", engine="fastparquet"
        )
        self.baseline_prediction_data = pd.read_parquet(
            self.file_path + "baseline.parquet", engine="fastparquet"
        )
        self.master = pd.concat(
            [self.final_data, self.baseline_prediction_data], axis=0, ignore_index=True
        )
        self.core_frames = [
            self.data,
            self.prediction_data,
            self.data_logits,
            self.final_data,
            self.final_pred_data,
            self.baseline_prediction_data,
        ]
        save_paths = [
            "train.parquet",
            "test.parquet",
            "data_with_ridit.hdfs",
            "final_data.parquet",
            "final_pred_data.parquet",
            "baseline.parquet",
        ]
        self.save_paths = [self.file_path + x for x in save_paths]
        self.core_names = [x.split(".")[0] for x in self.save_paths]
        self.final_data.rename(columns={"label": "target"}, inplace=True)
        self.data.rename(columns={"label": "target"}, inplace=True)

    @staticmethod
    def __save__(df: pd.DataFrame, loc: str):
        try:
            df.to_parquet(loc, engine="fastparquet", compression="brotli")
        except:
            return "Save Failed"
        return "Saved Successfully"

    def save_all(self):
        """
        Before Saving all objects ask question for each of them.
        And for each question if the answer is yes proceed to save otherwise continue.
        """
        exit_msg = "Exiting!"
        try:
            for df_name, df, df_loc in zip(
                self.core_names, self.core_frames, self.save_paths
            ):
                base_question = f"Do you want to save {df_name}?(Yes/No/Exit)"
                skip_msg = f"Skipping {df_name}"
                while True:
                    answer = input(base_question)
                    if answer == "Yes":
                        msg = self.__save__(df, df_loc)
                        print(df_name + msg)
                        break
                    elif answer in ["No", "n"]:
                        print(skip_msg)
                        break
                    elif answer in ["Exit", "e"]:
                        print(exit_msg)
                        return
                    else:
                        print("Not Valid Input")
                        continue
        except KeyboardInterrupt:
            print(exit_msg)
            return

    def categorise_data(self, df: pd.DataFrame = None):
        c_sel = ColumnSelectors()

        if isinstance(df, pd.DataFrame):
            ordinal_data = df.loc[:, c_sel.ordinal_cols]
            nominal_data = df.loc[:, c_sel.nominal_cols]
            binary_data = df.loc[:, c_sel.binary_cols]
            ratio_data = df.loc[:, c_sel.ratio_cols]
        else:
            df = self.final_data
            ordinal_data = df.loc[:, c_sel.ordinal_cols]
            nominal_data = df.loc[:, c_sel.nominal_cols]
            binary_data = df.loc[:, c_sel.binary_cols]
            ratio_data = df.loc[:, c_sel.ratio_cols]
        return ordinal_data, nominal_data, binary_data, ratio_data


if __name__ == "__main__":
    db = DFCollection()
