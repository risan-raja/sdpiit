import pandas as pd
from .components import ColumnSelectors as ColumnSelectors
from _typeshed import Incomplete

class DFCollection:
    file_path: str
    data: Incomplete
    prediction_data: Incomplete
    data_logits: Incomplete
    final_data: Incomplete
    final_pred_data: Incomplete
    baseline_prediction_data: Incomplete
    master: Incomplete
    core_frames: Incomplete
    save_paths: Incomplete
    core_names: Incomplete
    def __init__(self) -> None: ...
    @staticmethod
    def __save__(df: pd.DataFrame, loc: str): ...
    def save_all(self) -> None: ...
    def categorise_data(self, df: pd.DataFrame = ...): ...
