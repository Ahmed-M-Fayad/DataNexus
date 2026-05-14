import json

import pandas as pd

from .column_stats     import compute
from .pattern_detector import detect
from src.database      import get_db_session
from src.database.models import DataProfile


class ProfilerEngine:

    def profile(self, df: pd.DataFrame, dataset_id: int) -> DataProfile:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("[ProfilerEngine] Expected a pandas DataFrame.")
        if df.empty:
            raise ValueError(
                f"[ProfilerEngine] DataFrame for dataset_id={dataset_id} is empty — nothing to profile."
            )
        return self._save_profile(dataset_id, self._build_profile(df))

    def _build_profile(self, df: pd.DataFrame) -> dict:
        return {
            "row_count"   : len(df),
            "column_count": len(df.columns),
            "columns"     : {col: self._profile_column(df[col]) for col in df.columns},
        }

    def _profile_column(self, column: pd.Series) -> dict:
        stats = compute(column)
        return {
            "null_pct"        : stats["null_pct"],
            "distinct_count"  : stats["distinct_count"],
            "min"             : stats["min"],
            "max"             : stats["max"],
            "mean"            : stats["mean"],
            "median"          : stats["median"],
            "detected_pattern": detect(column),
        }

    def _save_profile(self, dataset_id: int, profile_dict: dict) -> DataProfile:
        record = DataProfile(
            dataset_id   = dataset_id,
            row_count    = profile_dict["row_count"],
            column_count = profile_dict["column_count"],
            profile_json = json.dumps(profile_dict, default=str),
        )
        with get_db_session() as session:
            session.add(record)
            session.flush()          # get DB-assigned id
            session.expunge(record)  # detach so the record survives session close
        return record
