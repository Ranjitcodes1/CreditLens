"""
Data Preparation, Cleaning, and Leakage Prevention Module for Loan Default Prediction.

This module provides reusable, leak-proof functions to:
1. Load raw Lending Club data safely with memory optimization.
2. Frame the credit risk classification problem by filtering closed loans
   and creating a binary target (1 = Charged Off/Default, 0 = Fully Paid).
3. Systematically eliminate target leakage and post-issuance features.
4. Clean, parse, and standardize data types (e.g. '36 months', '10+ years', '10.65%').
5. Handle missing values with justified domain strategies.
6. Export cleaned, production-ready data to data/processed.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Explicitly documented leakage columns
LEAKAGE_COLUMNS_REASONS: Dict[str, str] = {
    "funded_amnt": "Post-origination: Final amount funded by investors after credit evaluation.",
    "funded_amnt_inv": "Post-origination: Portion of funded amount committed by individual investors.",
    "out_prncp": "Post-issuance payment status: Remaining outstanding principal balance (0 for closed loans).",
    "out_prncp_inv": "Post-issuance payment status: Remaining outstanding principal for investors.",
    "total_pymnt": "Post-issuance target leak: Total payments received to date. If total_pymnt < funded_amnt, signals default.",
    "total_pymnt_inv": "Post-issuance target leak: Total payments received by investors to date.",
    "total_rec_prncp": "Post-issuance target leak: Principal received to date. Equal to loan_amnt for fully paid loans.",
    "total_rec_int": "Post-issuance target leak: Total interest received to date.",
    "total_rec_late_fee": "Post-issuance target leak: Late fees collected during loan repayment.",
    "recoveries": "Direct target leak: Gross post-charge-off recovery collections. If > 0, 100% defaulted.",
    "collection_recovery_fee": "Direct target leak: Fees paid to collection agencies to recover defaulted balances.",
    "last_pymnt_d": "Post-issuance event: Month the last payment was received.",
    "last_pymnt_amnt": "Post-issuance target leak: Last payment amount received before closure or default.",
    "last_credit_pull_d": "Post-issuance administrative: Most recent month Lending Club pulled credit for monitoring.",
    "loan_status": "Original raw target label column; replaced by binary 'target' column."
}

# Administrative, ID, free-text, and zero-variance columns to drop
ADMIN_AND_NOISE_COLUMNS: List[str] = [
    "id", "member_id", "url", "desc", "title", "emp_title", "zip_code",
    "pymnt_plan", "policy_code", "application_type", "initial_list_status",
    "acc_now_delinq", "delinq_amnt", "tax_liens", "collections_12_mths_ex_med",
    "chargeoff_within_12_mths"
]


def load_raw_data(
    filepath: Union[str, Path],
    nrows: Optional[int] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load raw loan dataset from disk with optional sampling."""
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
    """Filter out ongoing/active loans and define the binary default target."""
    if status_col not in df.columns:
        raise KeyError(f"Expected status column '{status_col}' not found in DataFrame.")

    initial_count = len(df)
    status_counts = df[status_col].value_counts(dropna=False).to_dict()

    paid_labels = ["Fully Paid"]
    default_labels = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]
    closed_labels = paid_labels + default_labels

    closed_mask = df[status_col].isin(closed_labels)
    df_closed = df[closed_mask].copy()
    dropped_count = initial_count - len(df_closed)

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

    return df_closed, summary


def clean_loan_data(
    df: pd.DataFrame,
    target_col: str = "target",
    missing_thresh: float = 0.50,
    save_processed_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Union[int, List, Dict]]]:
    """Clean, parse types, remove leakage columns, and export processed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe containing closed loans and target column.
    target_col : str, default='target'
        Name of the binary target column.
    missing_thresh : float, default=0.50
        Columns with a missing proportion greater than this threshold are dropped.
    save_processed_dir : str or Path, optional
        Directory path where 'loans_cleaned.parquet' and 'loans_cleaned.csv' will be written.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Cleaned dataframe and a comprehensive cleaning audit summary.
    """
    initial_shape = df.shape
    df_clean = df.copy()

    # 1. Identify columns with > 50% missing values
    missing_pct = df_clean.isnull().mean()
    high_missing_cols = missing_pct[missing_pct > missing_thresh].index.tolist()

    # 2. Identify leakage columns present
    leakage_cols_present = [c for c in LEAKAGE_COLUMNS_REASONS if c in df_clean.columns and c != target_col]

    # 3. Identify administrative / noise columns present
    admin_cols_present = [c for c in ADMIN_AND_NOISE_COLUMNS if c in df_clean.columns]

    # Combine all columns to drop
    cols_to_drop = list(set(high_missing_cols + leakage_cols_present + admin_cols_present))
    df_clean.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # 4. Data Type Parsing
    # Parse 'term' -> integer months (36 or 60)
    if "term" in df_clean.columns and df_clean["term"].dtype == object:
        df_clean["term"] = df_clean["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)

    # Parse 'int_rate' -> float percentage
    if "int_rate" in df_clean.columns and df_clean["int_rate"].dtype == object:
        df_clean["int_rate"] = df_clean["int_rate"].astype(str).str.rstrip("%").astype(float)

    # Parse 'revol_util' -> float percentage
    if "revol_util" in df_clean.columns and df_clean["revol_util"].dtype == object:
        df_clean["revol_util"] = df_clean["revol_util"].astype(str).str.rstrip("%").astype(float)

    # Parse 'emp_length' -> numeric years (0 to 10)
    emp_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10
    }
    if "emp_length" in df_clean.columns:
        df_clean["emp_length_years"] = df_clean["emp_length"].map(emp_map)
        # Create missingness indicator before imputation to capture potential non-random missingness
        df_clean["emp_length_missing"] = df_clean["emp_length_years"].isnull().astype(int)
        median_emp = float(df_clean["emp_length_years"].median())
        df_clean["emp_length_years"] = df_clean["emp_length_years"].fillna(median_emp)
        df_clean.drop(columns=["emp_length"], inplace=True)

    # 5. Missing value handling for low-missingness features
    # pub_rec_bankruptcies: missing indicates no recorded bankruptcies on bureau file
    if "pub_rec_bankruptcies" in df_clean.columns:
        df_clean["pub_rec_bankruptcies"] = df_clean["pub_rec_bankruptcies"].fillna(0.0)

    # revol_util: impute minor missing (<0.2%) with median
    if "revol_util" in df_clean.columns:
        median_util = float(df_clean["revol_util"].median())
        df_clean["revol_util"] = df_clean["revol_util"].fillna(median_util)

    # 6. Normalize Categoricals
    if "home_ownership" in df_clean.columns:
        # Collapse rare 'NONE' (3 cases) into 'OTHER'
        df_clean["home_ownership"] = df_clean["home_ownership"].replace({"NONE": "OTHER"})

    # 7. Deduplicate
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(df_clean)

    # 8. Export to data/processed if path provided
    if save_processed_dir:
        out_dir = Path(save_processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = out_dir / "loans_cleaned.parquet"
        csv_path = out_dir / "loans_cleaned.csv"

        df_clean.to_parquet(parquet_path, index=False)
        df_clean.to_csv(csv_path, index=False)
        print(f"Saved cleaned dataset to:\n  - {parquet_path.resolve()}\n  - {csv_path.resolve()}")

    summary = {
        "initial_shape": initial_shape,
        "final_shape": df_clean.shape,
        "high_missing_dropped_count": len(high_missing_cols),
        "leakage_dropped_count": len(leakage_cols_present),
        "admin_noise_dropped_count": len(admin_cols_present),
        "total_columns_dropped": len(cols_to_drop),
        "duplicates_removed": duplicates_removed,
        "retained_features": [c for c in df_clean.columns if c != target_col],
        "target_distribution": df_clean[target_col].value_counts().to_dict(),
    }

    return df_clean, summary


def inspect_data_health(
    df: pd.DataFrame,
    target_col: Optional[str] = "target",
) -> Dict[str, Union[int, float, Dict]]:
    """Generate diagnostic metrics on missing values, memory usage, and dtypes."""
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
