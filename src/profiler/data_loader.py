"""
data_loader.py
--------------
Loads data into a pandas DataFrame for the Data Profiler.

Two entry points:
    load_from_csv(file_path)          — load from a CSV file
    load_from_dataset_id(dataset_id)  — load from PostgreSQL by dataset_id
"""

import pandas as pd
from sqlalchemy import text, create_engine

from database.connection import get_db_session
from database.models import Dataset, DataSource


def load_from_csv(file_path: str, **csv_kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    file_path  : path to the CSV file
    csv_kwargs : any extra kwargs passed to pd.read_csv()

    Returns
    -------
    pandas DataFrame, or empty DataFrame on error.
    """
    try:
        df = pd.read_csv(file_path, **csv_kwargs)
        print(f"[data_loader] Loaded {len(df)} rows from '{file_path}'")
        return df
    except Exception as e:
        print(f"[data_loader] Error reading CSV '{file_path}': {e}")
        return pd.DataFrame()


def load_from_dataset_id(dataset_id: int, limit: int | None = None) -> pd.DataFrame:
    """
    Load a PostgreSQL table by dataset_id.

    Steps:
        1. Look up Dataset record → get schema_name + table_name
        2. Look up DataSource record → confirm it is active
        3. Run SELECT * (with optional LIMIT) and return as DataFrame

    Parameters
    ----------
    dataset_id : primary key from the datasets table
    limit      : optional row cap for large tables (uses SQL LIMIT)

    Returns
    -------
    pandas DataFrame with all rows from the table,
    or empty DataFrame if not found / error occurred.
    """
    try:
        with get_db_session() as session:

            dataset = session.query(Dataset).filter(
                Dataset.id        == dataset_id,
                Dataset.is_active == True,
            ).first()

            if dataset is None:
                print(f"[data_loader] No active dataset found with id={dataset_id}")
                return pd.DataFrame()

            source = session.query(DataSource).filter(
                DataSource.id        == dataset.source_id,
                DataSource.is_active == True,
            ).first()

            if source is None:
                print(f"[data_loader] No active source for dataset_id={dataset_id}")
                return pd.DataFrame()

            full_table = (
                f"{dataset.schema_name}.{dataset.table_name}"
                if dataset.schema_name
                else dataset.table_name
            )

            engine = create_engine(source.connection_string)

            if limit:
                query = text(f"SELECT * FROM {full_table} LIMIT :limit")
                df = pd.read_sql(query, engine, params={"limit": limit})
            else:
                query = text(f"SELECT * FROM {full_table}")
                df = pd.read_sql(query, engine)

            print(f"[data_loader] Loaded {len(df)} rows from '{full_table}' (dataset_id={dataset_id})")
            return df

    except Exception as e:
        print(f"[data_loader] Error loading dataset_id={dataset_id}: {e}")
        return pd.DataFrame()