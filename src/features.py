"""
Feature Engineering and Preprocessing Pipeline Module for Loan Default Prediction.

This module provides reusable, production-ready, scikit-learn compatible transformers:
1. DomainFeatureExtractor: Computes financial risk ratios, credit seasoning,
   ordinal ratings, and behavioral velocity flags without data leakage.
2. build_preprocessor: ColumnTransformer for scaling numerics and one-hot encoding categoricals.
3. build_feature_pipeline: Complete end-to-end pipeline encapsulating domain feature
   extraction and preprocessing.
4. engineer_features_df: Standalone DataFrame feature transformation for EDA and SHAP.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Subgrade and Grade ordinal maps
SUB_GRADES: List[str] = [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]
SUBGRADE_MAP: Dict[str, int] = {sg: idx + 1 for idx, sg in enumerate(SUB_GRADES)}
GRADE_MAP: Dict[str, int] = {g: idx + 1 for idx, g in enumerate("ABCDEFG")}

# Default categorical features for one-hot encoding
NOMINAL_CATEGORICAL_COLS: List[str] = [
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
]

# Baseline numerical features present prior to domain feature engineering
RAW_NUMERICAL_COLS: List[str] = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "pub_rec_bankruptcies",
    "emp_length_years",
    "emp_length_missing",
]

# Newly engineered domain features
ENGINEERED_FEATURE_NAMES: List[str] = [
    "credit_history_years",
    "loan_to_income",
    "installment_to_income",
    "revol_bal_to_income",
    "log_annual_inc",
    "sub_grade_ord",
    "grade_ord",
    "high_utilization_flag",
    "high_dti_flag",
    "velocity_hazard_flag",
    "derogatory_flag",
]


class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-Learn Transformer for retail lending domain feature engineering.

    Computes:
    - credit_history_years: Elapsed seasoning between earliest_cr_line and issue_d
    - loan_to_income: loan_amnt / (annual_inc + 1)
    - installment_to_income: (installment * 12) / (annual_inc + 1)
    - revol_bal_to_income: revol_bal / (annual_inc + 1)
    - log_annual_inc: log1p of annual_inc
    - sub_grade_ord: Monotonic ordinal score (1 to 35)
    - grade_ord: Monotonic ordinal score (1 to 7)
    - high_utilization_flag: Binary indicator if revol_util >= 80%
    - high_dti_flag: Binary indicator if dti >= 20%
    - velocity_hazard_flag: Binary indicator if inq_last_6mths >= 2
    - derogatory_flag: Binary indicator if pub_rec > 0 or pub_rec_bankruptcies > 0
    """

    def __init__(
        self,
        drop_original_dates: bool = True,
        drop_raw_grades: bool = True,
        century_cutoff_year: int = 2015,
    ):
        self.drop_original_dates = drop_original_dates
        self.drop_raw_grades = drop_raw_grades
        self.century_cutoff_year = century_cutoff_year
        self.median_cred_hist_: float = 12.5
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DomainFeatureExtractor":
        """Fit transformer by computing baseline training medians for missing values."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DomainFeatureExtractor requires a pandas DataFrame input.")

        self.feature_names_in_ = list(X.columns)

        if "issue_d" in X.columns and "earliest_cr_line" in X.columns:
            issue_dt = pd.to_datetime(X["issue_d"], format="%b-%y", errors="coerce")
            cr_dt = pd.to_datetime(X["earliest_cr_line"], format="%b-%y", errors="coerce")
            cr_dt = cr_dt.apply(
                lambda d: d.replace(year=d.year - 100)
                if pd.notnull(d) and d.year > self.century_cutoff_year
                else d
            )
            hist = (issue_dt - cr_dt).dt.days / 365.25
            median_hist = hist.median()
            self.median_cred_hist_ = float(median_hist) if pd.notnull(median_hist) else 12.5

        # Compute output feature names
        sample_transformed = self._transform_df(X.head(2))
        self.feature_names_out_ = list(sample_transformed.columns)

        return self

    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Internal transformation core returning a DataFrame."""
        X_out = X.copy()

        # 1. Credit History Length (Seasoning in years)
        if "issue_d" in X_out.columns and "earliest_cr_line" in X_out.columns:
            issue_dt = pd.to_datetime(X_out["issue_d"], format="%b-%y", errors="coerce")
            cr_dt = pd.to_datetime(X_out["earliest_cr_line"], format="%b-%y", errors="coerce")
            cr_dt = cr_dt.apply(
                lambda d: d.replace(year=d.year - 100)
                if pd.notnull(d) and d.year > self.century_cutoff_year
                else d
            )
            hist = (issue_dt - cr_dt).dt.days / 365.25
            X_out["credit_history_years"] = hist.fillna(self.median_cred_hist_).astype(float)
        else:
            X_out["credit_history_years"] = float(self.median_cred_hist_)

        # 2. Income and Burden Ratios
        if "annual_inc" in X_out.columns:
            inc = np.clip(X_out["annual_inc"].values, 0, None) + 1.0
            if "loan_amnt" in X_out.columns:
                X_out["loan_to_income"] = X_out["loan_amnt"].values / inc
            if "installment" in X_out.columns:
                X_out["installment_to_income"] = (X_out["installment"].values * 12.0) / inc
            if "revol_bal" in X_out.columns:
                X_out["revol_bal_to_income"] = X_out["revol_bal"].values / inc
            X_out["log_annual_inc"] = np.log1p(np.clip(X_out["annual_inc"].values, 0, None))

        # 3. Ordinal Rating Mappings
        if "sub_grade" in X_out.columns:
            X_out["sub_grade_ord"] = X_out["sub_grade"].map(SUBGRADE_MAP).fillna(18).astype(int)
        if "grade" in X_out.columns:
            X_out["grade_ord"] = X_out["grade"].map(GRADE_MAP).fillna(4).astype(int)

        # 4. Behavioral and Stress Indicators
        if "revol_util" in X_out.columns:
            X_out["high_utilization_flag"] = (X_out["revol_util"] >= 80.0).astype(int)
        if "dti" in X_out.columns:
            X_out["high_dti_flag"] = (X_out["dti"] >= 20.0).astype(int)
        if "inq_last_6mths" in X_out.columns:
            X_out["velocity_hazard_flag"] = (X_out["inq_last_6mths"] >= 2).astype(int)
        if "pub_rec" in X_out.columns and "pub_rec_bankruptcies" in X_out.columns:
            X_out["derogatory_flag"] = (
                (X_out["pub_rec"] > 0) | (X_out["pub_rec_bankruptcies"] > 0)
            ).astype(int)

        # Drop intermediate / non-numeric columns
        drop_cols: List[str] = []
        if self.drop_original_dates:
            drop_cols.extend(["issue_d", "earliest_cr_line"])
        if self.drop_raw_grades:
            drop_cols.extend(["grade", "sub_grade"])

        X_out.drop(columns=[c for c in drop_cols if c in X_out.columns], inplace=True)
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations to input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DomainFeatureExtractor requires a pandas DataFrame input.")
        return self._transform_df(X)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return output feature names after domain transformations."""
        return np.array(self.feature_names_out_, dtype=object)


def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to apply domain feature engineering directly to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned loan dataset.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with engineered financial metrics and ordinal ratings.
    """
    extractor = DomainFeatureExtractor(drop_original_dates=True, drop_raw_grades=True)
    return extractor.fit_transform(df)


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    scale_numeric: bool = True,
    sparse_output: bool = False,
    verbose_feature_names_out: bool = False,
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numerical and categorical features.

    Parameters
    ----------
    numerical_cols : List[str]
        List of numeric column names to scale or passthrough.
    categorical_cols : List[str], optional
        List of nominal categorical columns to one-hot encode. Defaults to NOMINAL_CATEGORICAL_COLS.
    scale_numeric : bool, default=True
        Whether to apply StandardScaler to numerical features. Set False for tree algorithms.
    sparse_output : bool, default=False
        Whether OneHotEncoder should produce sparse matrix.
    verbose_feature_names_out : bool, default=False
        Whether get_feature_names_out prefixes step names.

    Returns
    -------
    ColumnTransformer
        Configured scikit-learn preprocessor.
    """
    if categorical_cols is None:
        categorical_cols = NOMINAL_CATEGORICAL_COLS

    num_transformer = StandardScaler() if scale_numeric else "passthrough"
    cat_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=sparse_output,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        verbose_feature_names_out=verbose_feature_names_out,
    )

    return preprocessor


def build_feature_pipeline(
    scale_numeric: bool = True,
    drop_original_dates: bool = True,
    drop_raw_grades: bool = True,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[Pipeline, List[str]]:
    """Construct an end-to-end feature extraction and preprocessing Pipeline.

    Parameters
    ----------
    scale_numeric : bool, default=True
        If True, applies StandardScaler (ideal for Logistic Regression).
        If False, passes raw numerics (ideal for XGBoost / LightGBM).
    drop_original_dates : bool, default=True
        Whether to drop date strings after computing credit history length.
    drop_raw_grades : bool, default=True
        Whether to drop raw grade strings after computing monotonic ordinals.
    categorical_cols : List[str], optional
        Categorical features to encode.

    Returns
    -------
    Tuple[Pipeline, List[str]]
        Configured full pipeline and the list of intermediate engineered feature names.
    """
    if categorical_cols is None:
        categorical_cols = NOMINAL_CATEGORICAL_COLS

    # Determine numerical features post-domain extraction
    num_features = [
        c for c in RAW_NUMERICAL_COLS
        if c not in ["grade", "sub_grade", "issue_d", "earliest_cr_line"]
    ] + ENGINEERED_FEATURE_NAMES

    preprocessor = build_preprocessor(
        numerical_cols=num_features,
        categorical_cols=categorical_cols,
        scale_numeric=scale_numeric,
        sparse_output=False,
        verbose_feature_names_out=False,
    )

    full_pipeline = Pipeline([
        (
            "domain_features",
            DomainFeatureExtractor(
                drop_original_dates=drop_original_dates,
                drop_raw_grades=drop_raw_grades,
            ),
        ),
        ("preprocessor", preprocessor),
    ])

    return full_pipeline, num_features


def get_feature_groups() -> Dict[str, List[str]]:
    """Return catalog of features grouped by modeling and domain role."""
    return {
        "raw_numerical": RAW_NUMERICAL_COLS,
        "nominal_categorical": NOMINAL_CATEGORICAL_COLS,
        "engineered_metrics": ENGINEERED_FEATURE_NAMES,
        "ordinal_ratings": ["grade_ord", "sub_grade_ord"],
        "binary_hazard_flags": [
            "high_utilization_flag",
            "high_dti_flag",
            "velocity_hazard_flag",
            "derogatory_flag",
            "emp_length_missing",
        ],
    }
