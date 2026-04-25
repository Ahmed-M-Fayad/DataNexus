"""
profiler_engine.py
------------------
Main entry point for the Data Profiler (Step 3).

The ValidationEngine calls this after it has already loaded the dataset
into a pandas DataFrame. The profiler never fetches data itself.

Usage (inside ValidationEngine.run)
-------------------------------------
    from profiler import ProfilerEngine

    engine = ProfilerEngine()
    data_profile = engine.profile(df, dataset_id=1)
    # data_profile is now a saved DataProfile ORM object

Required profile_json structure stored in data_profiles
---------------------------------------------------------
{
    "row_count":    int,
    "column_count": int,
    "columns": {
        "<col_name>": {
            "null_pct":        float,
            "distinct_count":  int,
            "min":             number | null,
            "max":             number | null,
            "mean":            number | null,
            "median":          number | null,
            "detected_pattern": string | null   e.g. "email", "phone", "date"
        }
    }
}
"""

import json

import pandas as pd

from .column_stats      import compute
from .pattern_detector  import detect

from database.connection import get_db_session
from database.models     import DataProfile


class ProfilerEngine:
    """
    Accepts a pandas DataFrame that the ValidationEngine has already loaded,
    builds a statistical profile, saves it to data_profiles, and returns the
    saved DataProfile ORM object.
    """

    def profile(self, df: pd.DataFrame, dataset_id: int) -> DataProfile:
        """
        Profile a DataFrame and persist the result to the database.

        Parameters
        ----------
        df         : the dataset to profile, loaded by the ValidationEngine
        dataset_id : FK to datasets.id — identifies which dataset this snapshot belongs to

        Returns
        -------
        DataProfile ORM object with all fields populated (including the DB-assigned id).

        Raises
        ------
        TypeError  if df is not a pandas DataFrame
        ValueError if df is empty
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("[ProfilerEngine] Expected a pandas DataFrame.")
        if df.empty:
            raise ValueError(
                f"[ProfilerEngine] DataFrame for dataset_id={dataset_id} is empty "
                "— nothing to profile."
            )

        profile_dict = self._build_profile(df)
        return self._save_profile(dataset_id, profile_dict)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_profile(self, df: pd.DataFrame) -> dict:
        """Iterate over all columns and assemble the full profile dict."""
        return {
            "row_count"   : len(df),
            "column_count": len(df.columns),
            "columns"     : {
                col_name: self._profile_column(df[col_name])
                for col_name in df.columns
            },
        }

    def _profile_column(self, column: pd.Series) -> dict:
        """
        Build the profile dict for a single column.

        Calls column_stats.compute() for numeric/null statistics,
        then pattern_detector.detect() for string pattern recognition.
        Both results are merged into one flat dict per the JSON contract.
        """
        stats           = compute(column)
        detected_pattern = detect(column)

        return {
            "null_pct"        : stats["null_pct"],
            "distinct_count"  : stats["distinct_count"],
            "min"             : stats["min"],
            "max"             : stats["max"],
            "mean"            : stats["mean"],
            "median"          : stats["median"],
            "detected_pattern": detected_pattern,   # str like "email" / "phone", or None
        }

    def _save_profile(self, dataset_id: int, profile_dict: dict) -> DataProfile:
        """
        Insert a DataProfile row and return the populated ORM object.

        Uses session.flush() inside the context manager so the DB assigns an id,
        then expunge() detaches the object cleanly before the session closes.
        The context manager commits the transaction on exit.
        """
        record = DataProfile(
            dataset_id   = dataset_id,
            row_count    = profile_dict["row_count"],
            column_count = profile_dict["column_count"],
            profile_json = json.dumps(profile_dict, default=str),
        )

        with get_db_session() as session:
            session.add(record)
            session.flush()          # sends INSERT → DB assigns record.id
            session.expunge(record)  # detach before session closes — keeps record usable

        return record  # has id, all columns populated