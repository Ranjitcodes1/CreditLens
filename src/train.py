"""
Model Training, Cross-Validation, Cost Optimization, and Serialization Module.

This module provides an end-to-end production pipeline to:
1. Load cleaned data and perform stratified train/test splitting.
2. Benchmark candidate models (Logistic Regression, Random Forest, LightGBM, XGBoost)
   across Stratified K-Fold CV on ROC-AUC, PR-AUC, and Brier Score.
3. Perform Bayesian hyperparameter optimization using Optuna.
4. Calculate financial cost-sensitive optimal decision thresholds.
5. Serialize the champion end-to-end Pipeline and underwriting metadata to models/.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

from src.features import (
    DomainFeatureExtractor,
    build_feature_pipeline,
    build_preprocessor,
    RAW_NUMERICAL_COLS,
    NOMINAL_CATEGORICAL_COLS,
    ENGINEERED_FEATURE_NAMES,
)

# Silence Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_train_test_data(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load cleaned dataset and generate stratified train and test partitions."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {path.resolve()}")

    df = pd.read_parquet(path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Loaded dataset: {len(df):,} total records.")
    print(f"Train set: {len(X_train):,} samples (Default rate: {y_train.mean()*100:.2f}%)")
    print(f"Test set:  {len(X_test):,} samples (Default rate: {y_test.mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test


def evaluate_cv_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Benchmark candidate credit default classification models across Stratified K-Fold CV.

    Evaluates:
    - Logistic Regression (with scaled numerics & balanced weights)
    - Random Forest (balanced weights)
    - LightGBM (class_weight='balanced')
    - XGBoost (scale_pos_weight)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    # Build scaled and unscaled preprocessing pipelines
    pipe_scaled, _ = build_feature_pipeline(scale_numeric=True)
    pipe_unscaled, _ = build_feature_pipeline(scale_numeric=False)

    print("Fitting feature extraction & preprocessing pipelines on training data...")
    X_train_scaled = pipe_scaled.fit_transform(X_train)
    X_train_unscaled = pipe_unscaled.fit_transform(X_train)
    y_arr = y_train.values

    model_configs: Dict[str, Tuple[Any, np.ndarray]] = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
            X_train_scaled,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=150, max_depth=10, class_weight="balanced",
                n_jobs=-1, random_state=random_state
            ),
            X_train_unscaled,
        ),
        "LightGBM": (
            lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.03, class_weight="balanced",
                random_state=random_state, verbose=-1
            ),
            X_train_unscaled,
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                n_estimators=150, learning_rate=0.03, scale_pos_weight=pos_weight,
                tree_method="hist", random_state=random_state, eval_metric="logloss"
            ),
            X_train_unscaled,
        ),
    }

    results: List[Dict[str, Any]] = []

    print(f"\nBenchmarking {len(model_configs)} models with {n_splits}-fold Stratified CV...")
    for model_name, (clf, X_mat) in model_configs.items():
        roc_scores: List[float] = []
        pr_scores: List[float] = []
        brier_scores: List[float] = []
        f1_scores: List[float] = []
        recall_scores: List[float] = []
        precision_scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_arr), 1):
            clf.fit(X_mat[train_idx], y_arr[train_idx])
            probs = clf.predict_proba(X_mat[val_idx])[:, 1]
            preds = (probs >= 0.50).astype(int)

            roc_scores.append(roc_auc_score(y_arr[val_idx], probs))
            pr_scores.append(average_precision_score(y_arr[val_idx], probs))
            brier_scores.append(brier_score_loss(y_arr[val_idx], probs))
            f1_scores.append(f1_score(y_arr[val_idx], preds, zero_division=0))
            recall_scores.append(recall_score(y_arr[val_idx], preds, zero_division=0))
            precision_scores.append(precision_score(y_arr[val_idx], preds, zero_division=0))

        results.append({
            "Model": model_name,
            "ROC-AUC (Mean)": np.mean(roc_scores),
            "ROC-AUC (Std)": np.std(roc_scores),
            "PR-AUC (Mean)": np.mean(pr_scores),
            "PR-AUC (Std)": np.std(pr_scores),
            "Brier Score": np.mean(brier_scores),
            "Recall@0.50": np.mean(recall_scores),
            "Precision@0.50": np.mean(precision_scores),
            "F1@0.50": np.mean(f1_scores),
        })
        print(f"  [OK] {model_name:<20} | ROC-AUC: {np.mean(roc_scores):.4f} | PR-AUC: {np.mean(pr_scores):.4f} | Brier: {np.mean(brier_scores):.4f}")

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC (Mean)", ascending=False).reset_index(drop=True)
    return results_df


def optimize_threshold_cost(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    c_fn: float = 5000.0,
    c_fp: float = 500.0,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate cost-sensitive optimal decision threshold minimizing financial loss.

    Parameters
    ----------
    y_true : np.ndarray
        True binary target array (0 = Paid, 1 = Default).
    y_probs : np.ndarray
        Predicted probabilities of default.
    c_fn : float, default=5000.0
        Cost of False Negative (underwriting a loan that defaults).
    c_fp : float, default=500.0
        Cost of False Positive (rejecting an applicant who would have fully paid).
    thresholds : np.ndarray, optional
        Array of candidate threshold values. Defaults to np.linspace(0.05, 0.95, 91).

    Returns
    -------
    Dict[str, Any]
        Optimal threshold, minimum aggregate cost, naive baseline cost, and full cost curve.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    costs = []
    recalls = []
    precisions = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_cost = (fn * c_fn) + (fp * c_fp)
        costs.append(total_cost)
        recalls.append(tp / max(1, (tp + fn)))
        precisions.append(tp / max(1, (tp + fp)))

    costs = np.array(costs)
    best_idx = int(np.argmin(costs))
    optimal_threshold = float(thresholds[best_idx])
    min_cost = float(costs[best_idx])

    # Cost at standard naive 0.50 threshold
    y_pred_naive = (y_probs >= 0.50).astype(int)
    tn_n, fp_n, fn_n, tp_n = confusion_matrix(y_true, y_pred_naive, labels=[0, 1]).ravel()
    cost_naive = float((fn_n * c_fn) + (fp_n * c_fp))

    # Cost of unhedged baseline: approve every loan (threshold = 1.0)
    total_defaults = int((y_true == 1).sum())
    cost_approve_all = float(total_defaults * c_fn)

    savings_vs_approve_all = float(cost_approve_all - min_cost)
    savings_vs_naive = float(cost_naive - min_cost)

    return {
        "optimal_threshold": round(optimal_threshold, 3),
        "min_cost": round(min_cost, 2),
        "cost_naive_050": round(cost_naive, 2),
        "cost_approve_all": round(cost_approve_all, 2),
        "savings_vs_approve_all": round(savings_vs_approve_all, 2),
        "savings_vs_naive": round(savings_vs_naive, 2),
        "savings_pct_vs_approve_all": round((savings_vs_approve_all / cost_approve_all) * 100, 2),
        "thresholds": thresholds,
        "costs": costs,
        "recalls": np.array(recalls),
        "precisions": np.array(precisions),
    }


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 25,
    n_splits: int = 3,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], float]:
    """Bayesian hyperparameter optimization for LightGBM using Optuna."""
    print(f"\nLaunching Optuna Bayesian tuning for LightGBM ({n_trials} trials, {n_splits}-fold CV)...")
    pipe_unscaled, _ = build_feature_pipeline(scale_numeric=False)
    X_train_proc = pipe_unscaled.fit_transform(X_train)
    y_arr = y_train.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100, step=10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

        pr_scores = []
        for train_idx, val_idx in skf.split(X_train_proc, y_arr):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train_proc[train_idx], y_arr[train_idx])
            probs = clf.predict_proba(X_train_proc[val_idx])[:, 1]
            pr_scores.append(average_precision_score(y_arr[val_idx], probs))

        return float(np.mean(pr_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = random_state
    best_params["verbose"] = -1
    best_params["n_jobs"] = -1

    best_pr_auc = float(study.best_value)
    print(f"Optuna tuning finished. Best CV PR-AUC: {best_pr_auc:.4f}")
    print(f"Optimal hyperparameters: {best_params}")

    return best_params, best_pr_auc


def train_and_save_champion_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Optional[Dict[str, Any]] = None,
    save_path: Optional[Union[str, Path]] = None,
    c_fn: float = 5000.0,
    c_fp: float = 500.0,
) -> Dict[str, Any]:
    """Train the champion model on the complete training set, optimize threshold, and serialize."""
    if save_path is None:
        save_path = DEFAULT_MODEL_DIR / "loan_default_model.joblib"
    else:
        save_path = Path(save_path)

    if best_params is None:
        best_params = {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 40,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

    # Construct the full production end-to-end Pipeline
    num_features = [
        c for c in RAW_NUMERICAL_COLS
        if c not in ["grade", "sub_grade", "issue_d", "earliest_cr_line"]
    ] + ENGINEERED_FEATURE_NAMES

    preprocessor = build_preprocessor(
        numerical_cols=num_features,
        categorical_cols=NOMINAL_CATEGORICAL_COLS,
        scale_numeric=False,  # Tree models don't require feature scaling
        sparse_output=False,
        verbose_feature_names_out=False,
    )

    champion_classifier = lgb.LGBMClassifier(**best_params)

    production_pipeline = Pipeline([
        ("domain_features", DomainFeatureExtractor(drop_original_dates=True, drop_raw_grades=True)),
        ("preprocessor", preprocessor),
        ("classifier", champion_classifier),
    ])

    print("\nFitting production pipeline (DomainFeatureExtractor -> Preprocessor -> LightGBM)...")
    production_pipeline.fit(X_train, y_train)

    # Test set evaluation
    y_test_arr = y_test.values
    test_probs = production_pipeline.predict_proba(X_test)[:, 1]

    test_roc_auc = float(roc_auc_score(y_test_arr, test_probs))
    test_pr_auc = float(average_precision_score(y_test_arr, test_probs))
    test_brier = float(brier_score_loss(y_test_arr, test_probs))

    # Cost-sensitive threshold evaluation on test set
    cost_analysis = optimize_threshold_cost(y_test_arr, test_probs, c_fn=c_fn, c_fp=c_fp)
    opt_t = cost_analysis["optimal_threshold"]

    test_preds_opt = (test_probs >= opt_t).astype(int)
    test_recall_opt = float(recall_score(y_test_arr, test_preds_opt))
    test_precision_opt = float(precision_score(y_test_arr, test_preds_opt))
    test_f1_opt = float(f1_score(y_test_arr, test_preds_opt))

    print("\n=== HELD-OUT TEST SET EVALUATION ===")
    print(f"ROC-AUC:               {test_roc_auc:.4f}")
    print(f"PR-AUC:                {test_pr_auc:.4f}")
    print(f"Brier Score:           {test_brier:.4f}")
    print(f"Optimal Threshold:     {opt_t:.3f}")
    print(f"Recall at Optimal:     {test_recall_opt*100:.2f}%")
    print(f"Precision at Optimal:  {test_precision_opt*100:.2f}%")
    print(f"F1 at Optimal:         {test_f1_opt:.4f}")
    print(f"Aggregate Cost:        ${cost_analysis['min_cost']:,.2f}")
    print(f"Savings vs Baseline:   ${cost_analysis['savings_vs_approve_all']:,.2f} ({cost_analysis['savings_pct_vs_approve_all']}%)")

    # Extract feature names from preprocessor
    fitted_preprocessor = production_pipeline.named_steps["preprocessor"]
    transformed_feature_names = list(fitted_preprocessor.get_feature_names_out())

    # Build production bundle
    artifact_bundle = {
        "pipeline": production_pipeline,
        "optimal_threshold": opt_t,
        "cost_fn": c_fn,
        "cost_fp": c_fp,
        "test_metrics": {
            "roc_auc": test_roc_auc,
            "pr_auc": test_pr_auc,
            "brier_score": test_brier,
            "recall": test_recall_opt,
            "precision": test_precision_opt,
            "f1": test_f1_opt,
        },
        "cost_analysis": {
            "min_cost": cost_analysis["min_cost"],
            "cost_naive_050": cost_analysis["cost_naive_050"],
            "cost_approve_all": cost_analysis["cost_approve_all"],
            "savings_vs_approve_all": cost_analysis["savings_vs_approve_all"],
            "savings_pct_vs_approve_all": cost_analysis["savings_pct_vs_approve_all"],
        },
        "feature_names": transformed_feature_names,
        "best_params": best_params,
        "model_type": "LightGBM Classifier",
        "created_at": datetime.now().isoformat(),
    }

    joblib.dump(artifact_bundle, save_path)
    print(f"\nSuccessfully serialized production bundle to: {save_path.resolve()}")
    return artifact_bundle


def main():
    parser = argparse.ArgumentParser(description="Train and optimize loan default prediction model.")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter optimization.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna tuning trials.")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_train_test_data()

    # Step 1: Model Benchmarking
    cv_summary = evaluate_cv_models(X_train, y_train, n_splits=5)
    print("\n=== 5-FOLD CV BENCHMARK SUMMARY ===")
    print(cv_summary.to_string(index=False))

    # Step 2: Optuna Tuning
    best_params = None
    if args.tune:
        best_params, _ = tune_lightgbm(X_train, y_train, n_trials=args.trials, n_splits=3)

    # Step 3: Train & Save Champion Pipeline
    train_and_save_champion_model(X_train, y_train, X_test, y_test, best_params=best_params)


if __name__ == "__main__":
    main()
