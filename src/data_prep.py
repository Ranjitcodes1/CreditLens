"""
Data Preparation and Target Definition Module for Loan Default Prediction.

This module provides reusable, leak-proof functions to:
1. Load raw Lending Club data safely with memory optimization.
2. Frame the credit risk classification problem by filtering closed loans
   and creating a binary target (1 = Charged Off/Default, 0 = Fully Paid).
3. Inspect dataset health, missingness, and class imbalance distributions.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def load_raw_data(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load raw loan dataset from disk with optional sampling.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw CSV or compressed file.
    nrows : int, optional
        Number of rows to read from the top of the file.
    sample_size : int, optional
        If the loaded dataset exceeds sample_size, take a random sample
        of this size using random_state for reproducibility.
    random_state : int, default=42
        Random seed for sampling.

    Returns
    -------
    pd.DataFrame
        Loaded loan dataframe.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")

    print(f"Loading raw loan data from: {path.name}...")
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    print(f"Loaded raw records: {len(df):,} rows x {len(df.columns)} columns.")

    if sample_size and len(df) > sample_size:
        print(f"Subsampling to {sample_size:,} records with random_state={random_state}...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Subsampled shape: {df.shape}")

    return df


def filter_closed_loans_and_define_target(
    df: pd.DataFrame,
    status_col: str = "loan_status",
    target_col: str = "target",
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """Filter out ongoing/active loans and define the binary default target.

    Business Rationale:
    -------------------
    In credit risk modeling, loans marked as 'Current', 'In Grace Period', or
    'Late (16-30 days)' have not reached their terminal outcome. We do not know
    whether they will eventually fully repay or charge off (a phenomenon known
    in survival analysis and credit scoring as 'right-censoring').
    
    Training a binary classifier on in-progress loans introduces ground-truth
    label noise. Therefore, we restrict our universe to matured/terminal loans:
      - 'Fully Paid' -> 0 (Good borrower)
      - 'Charged Off' or 'Default' -> 1 (Bad borrower / Default)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the loan status column.
    status_col : str, default='loan_status'
        Name of the column containing loan outcome statuses.
    target_col : str, default='target'
        Name of the target column to create.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Filtered dataframe with target column added, and a summary dictionary.
    """
    if status_col not in df.columns:
        raise KeyError(f"Expected status column '{status_col}' not found in DataFrame.")

    initial_count = len(df)
    status_counts = df[status_col].value_counts(dropna=False).to_dict()

    # Define terminal outcomes
    paid_labels = ["Fully Paid"]
    default_labels = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]
    closed_labels = paid_labels + default_labels

    # Drop in-progress / unfinalized loans
    closed_mask = df[status_col].isin(closed_labels)
    df_closed = df[closed_mask].copy()
    dropped_count = initial_count - len(df_closed)

    # Assign binary target
    df_closed[target_col] = df_closed[status_col].isin(default_labels).astype(int)

    default_count = int(df_closed[target_col].sum())
    paid_count = int((df_closed[target_col] == 0).sum())
    default_rate = float(df_closed[target_col].mean())

    summary = {
        "initial_rows": initial_count,
        "dropped_in_progress": dropped_count,
        "final_closed_rows": len(df_closed),
        "fully_paid_count": paid_count,
        "default_count": default_count,
        "default_rate_pct": round(default_rate * 100, 2),
        "status_breakdown": status_counts,
    }

    print("\n--- Problem Framing & Target Definition Summary ---")
    print(f"Total raw loans evaluated:     {initial_count:,}")
    print(f"In-progress loans dropped:     {dropped_count:,} (right-censored)")
    print(f"Terminal loans retained:       {len(df_closed):,}")
    print(f"  Class 0 (Fully Paid):        {paid_count:,} ({100 - summary['default_rate_pct']:.2f}%)")
    print(f"  Class 1 (Default/Charged):   {default_count:,} ({summary['default_rate_pct']:.2f}%)")
    print(f"Class imbalance ratio:         ~{paid_count / max(1, default_count):.2f}:1")

    return df_closed, summary


def inspect_data_health(
    df: pd.DataFrame,
    target_col: Optional[str] = "target",
) -> Dict[str, Union[int, float, Dict]]:
    """Generate diagnostic metrics on missing values, memory usage, and dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to inspect.
    target_col : str, optional
        Target column name for balance inspection.

    Returns
    -------
    Dict
        Dictionary containing comprehensive data health metrics.
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_series = df.isnull().sum()
    missing_pct_series = (missing_series / len(df)) * 100

    cols_100_missing = missing_pct_series[missing_pct_series == 100].index.tolist()
    cols_gt50_missing = missing_pct_series[(missing_pct_series > 50) & (missing_pct_series < 100)].index.tolist()
    cols_gt0_missing = missing_pct_series[(missing_pct_series > 0) & (missing_pct_series <= 50)].index.tolist()
    cols_0_missing = missing_pct_series[missing_pct_series == 0].index.tolist()

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    health_report = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "memory_usage_mb": round(float(memory_mb), 2),
        "total_missing_cells": int(missing_series.sum()),
        "total_missing_pct": round(float((missing_series.sum() / max(1, total_cells)) * 100), 2),
        "cols_100_missing_count": len(cols_100_missing),
        "cols_gt50_missing_count": len(cols_gt50_missing),
        "cols_gt0_le50_missing_count": len(cols_gt0_missing),
        "cols_complete_count": len(cols_0_missing),
        "dtype_counts": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
    }

    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
        proportions = df[target_col].value_counts(normalize=True).round(4).to_dict()
        health_report["class_balance"] = {
            "counts": {int(k): int(v) for k, v in counts.items()},
            "proportions": {int(k): float(v) for k, v in proportions.items()},
        }

    return health_report
