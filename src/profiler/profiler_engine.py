"""
profiler_engine.py
------------------
Main entry point for the Data Profiler (Step 3).

Usage
-----
    from profiler_engine import ProfilerEngine

    engine = ProfilerEngine()

    # from a CSV file
    result = engine.profile_csv("data/customers.csv")

    # from a PostgreSQL table (dataset_id is all you need)
    result = engine.profile_table(dataset_id=1)

Required output structure per column
-------------------------------------
{
    "row_count":    int,
    "column_count": int,
    "columns": {
        "<col_name>": {
            "null_pct":     float,
            "unique_count": int,
            "min":          number | null,
            "max":          number | null,
            "mean":         number | null,
            "median":       number | null,
            "pattern": {
                "pattern":  string | null,
                "rate":     float  | null
            }
        }
    }
}
"""

import json
from datetime import datetime
import pandas as pd

from data_loader import load_from_csv, load_from_dataset_id
from column_stats import basic_column_info, numeric_column_stats, detect_outliers
from pattern_detector import pattern_detector
from database.connection import get_db_session
from database.models import DataProfile


class ProfilerEngine:
    """
    Scans a pandas DataFrame and returns a structured profile dict
    that matches the required JSON contract exactly.
    """

    def profile_csv(self, file_path: str, **csv_kwargs) -> dict:
        """Profile a CSV file. Returns the profile dict."""
        df = load_from_csv(file_path, **csv_kwargs)
        return self._build_profile(df)

    def profile_table(self,
                      dataset_id: int,
                      limit: int | None = None,
                      save_to_db: bool = True) -> dict:
        """
        Profile a PostgreSQL table by dataset_id.

        Looks up schema + table name automatically from the datasets table,
        loads the data, profiles it, and optionally saves to data_profiles.

        Parameters
        ----------
        dataset_id : int   — FK to datasets.id
        limit      : int   — optional row cap for large tables
        save_to_db : bool  — persist result to data_profiles table
        """
        df      = load_from_dataset_id(dataset_id, limit=limit)
        profile = self._build_profile(df)

        if save_to_db:
            self._save_profile(dataset_id, profile)

        return profile

    def profile_dataframe(self, df: pd.DataFrame) -> dict:
        """Profile an already-loaded DataFrame directly."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to profile.")
        return self._build_profile(df)


    def _build_profile(self, df: pd.DataFrame) -> dict:
        """Iterate over all columns and assemble the final profile dict."""
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
        Build the profile for a single column.

        Merges basic stats + numeric stats flat into the dict,
        then nests pattern detection under the "pattern" key.
        Outliers are included as an extra key for deeper analysis
        but are not part of the required contract.
        """
        basic   = basic_column_info(column)
        numeric = numeric_column_stats(column)
        pattern = pattern_detector(column)
        outliers = detect_outliers(column)

        return {
            "null_pct"    : basic["null_pct"],
            "unique_count": basic["unique_count"],
            "min"         : numeric["min"],
            "max"         : numeric["max"],
            "mean"        : numeric["mean"],
            "median"      : numeric["median"],
            "pattern"     : pattern,
            "outliers"    : outliers,
        }

    def _save_profile(self, dataset_id: int, profile: dict) -> None:
        """Insert a DataProfile row into the database."""
        with get_db_session() as session:
            session.add(DataProfile(
                dataset_id   = dataset_id,
                row_count    = profile["row_count"],
                column_count = profile["column_count"],
                profile_json = json.dumps(profile, default=str),
            ))
