"""
Inference Scoring and SHAP Explainability Engine for Loan Default Prediction.

This module provides production-ready utilities to:
1. Load the serialized champion pipeline and underwriting metadata.
2. Predict default probability and generate automated underwriting decisions (ACCEPT/REJECT).
3. Compute expected financial credit loss.
4. Generate local SHAP feature attributions and regulatory adverse action reason codes
   compliant with Fair Credit Reporting Act (FCRA) standards.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import shap

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "loan_default_model.joblib"

# Regulatory Adverse Action Reason Mappings (ECOA / FCRA standard reasons)
ADVERSE_ACTION_REASONS: Dict[str, str] = {
    "int_rate": "High interest rate reflecting elevated assigned market risk grade",
    "sub_grade_ord": "Subprime credit risk score assigned during credit evaluation",
    "grade_ord": "Low credit bureau rating grade",
    "loan_to_income": "Requested loan amount is high relative to verified annual income",
    "installment_to_income": "Monthly loan installment represents excessive burden relative to gross income",
    "revol_bal_to_income": "Excessive revolving balance relative to annual income",
    "dti": "High debt-to-income ratio (non-mortgage monthly debt commitments)",
    "high_dti_flag": "Debt-to-income ratio exceeds underwriting threshold (>= 20%)",
    "revol_util": "High revolving credit line utilization rate (credit lines near maximum capacity)",
    "high_utilization_flag": "Revolving credit line utilization exceeds high-risk threshold (>= 80%)",
    "inq_last_6mths": "Multiple credit inquiries within past 6 months (credit-seeking velocity)",
    "velocity_hazard_flag": "Elevated credit application velocity in recent bureau reporting",
    "term": "Extended loan maturity duration (60-month term carries elevated default volatility)",
    "derogatory_flag": "Presence of derogatory public records or historical bankruptcy filings",
    "pub_rec": "Derogatory public record on credit bureau file",
    "pub_rec_bankruptcies": "Recorded history of legal bankruptcy filing",
    "annual_inc": "Insufficient annual gross earnings to support credit line",
    "log_annual_inc": "Low verified income relative to credit obligations",
    "credit_history_years": "Short credit file duration / limited credit seasoning",
    "delinq_2yrs": "Past delinquency incidents (30+ days past due) within the last 24 months",
    "open_acc": "Low number of established open credit trade lines",
    "total_acc": "Limited total historical credit file depth",
}


def load_production_model(model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load the serialized champion model bundle from disk."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path.resolve()}")

    bundle = joblib.load(path)
    return bundle


def predict_loan_applicant(
    applicant_data: Union[Dict[str, Any], pd.DataFrame],
    model_bundle: Optional[Dict[str, Any]] = None,
    custom_threshold: Optional[float] = None,
    lgd: float = 0.60,
) -> Dict[str, Any]:
    """Score a loan applicant and generate underwriting decisions.

    Parameters
    ----------
    applicant_data : dict or pd.DataFrame
        Applicant credit and financial attributes.
    model_bundle : dict, optional
        Loaded model bundle. If None, loaded from DEFAULT_MODEL_PATH.
    custom_threshold : float, optional
        Underwriting decision cutoff. Defaults to optimal_threshold from model bundle.
    lgd : float, default=0.60
        Loss Given Default rate (assumed 60% loss upon charge-off).

    Returns
    -------
    dict
        Scoring result containing probability, decision, risk tier, and expected loss.
    """
    if model_bundle is None:
        model_bundle = load_production_model()

    pipeline = model_bundle["pipeline"]
    optimal_threshold = model_bundle.get("optimal_threshold", 0.370)
    decision_threshold = custom_threshold if custom_threshold is not None else optimal_threshold

    # Convert dict to DataFrame if needed
    if isinstance(applicant_data, dict):
        df_app = pd.DataFrame([applicant_data])
    else:
        df_app = applicant_data.copy()

    # Predict default probability
    p_default = float(pipeline.predict_proba(df_app)[0, 1])

    # Underwriting Decision
    is_rejected = p_default >= decision_threshold
    decision = "REJECT" if is_rejected else "ACCEPT"

    # Risk Tier Categorization
    if p_default < 0.15:
        risk_tier = "Tier 1: Prime (Low Risk)"
        risk_color = "#2ecc71"
    elif p_default < 0.30:
        risk_tier = "Tier 2: Near-Prime (Moderate Risk)"
        risk_color = "#3498db"
    elif p_default < 0.45:
        risk_tier = "Tier 3: Subprime (Elevated Risk)"
        risk_color = "#e67e22"
    else:
        risk_tier = "Tier 4: Deep Subprime (High Risk)"
        risk_color = "#e74c3c"

    # Calculate Expected Dollar Loss (EL = PD * EAD * LGD)
    loan_amount = float(df_app["loan_amnt"].values[0]) if "loan_amnt" in df_app.columns else 10000.0
    expected_loss = p_default * loan_amount * lgd

    return {
        "default_probability": round(p_default, 4),
        "default_probability_pct": round(p_default * 100, 2),
        "decision": decision,
        "decision_threshold": round(decision_threshold, 3),
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "loan_amount": loan_amount,
        "expected_loss": round(expected_loss, 2),
        "applicant_input": df_app.iloc[0].to_dict(),
    }


def explain_applicant_shap(
    applicant_data: Union[Dict[str, Any], pd.DataFrame],
    model_bundle: Optional[Dict[str, Any]] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Compute local SHAP feature attributions and generate FCRA-compliant adverse action reasons.

    Parameters
    ----------
    applicant_data : dict or pd.DataFrame
        Single applicant record.
    model_bundle : dict, optional
        Loaded model bundle.
    top_n : int, default=5
        Number of top risk-increasing factors to return.

    Returns
    -------
    dict
        Local SHAP explanation including positive contributors, negative contributors,
        and human-readable adverse action reasons.
    """
    if model_bundle is None:
        model_bundle = load_production_model()

    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle["feature_names"]

    if isinstance(applicant_data, dict):
        df_app = pd.DataFrame([applicant_data])
    else:
        df_app = applicant_data.copy()

    # Deconstruct pipeline steps to access preprocessed vector and classifier
    fe_step = pipeline.named_steps["domain_features"]
    prep_step = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform applicant record through preprocessing
    X_fe = fe_step.transform(df_app)
    X_prep = prep_step.transform(X_fe)

    # Compute Tree SHAP values
    explainer = shap.TreeExplainer(classifier)
    raw_shap = explainer.shap_values(X_prep)

    # Handle shape (array or list of arrays)
    if isinstance(raw_shap, list):
        shap_vec = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
    elif raw_shap.ndim == 2:
        shap_vec = raw_shap[0]
    elif raw_shap.ndim == 3:
        shap_vec = raw_shap[0, :, 1]
    else:
        shap_vec = raw_shap.flatten()

    base_value = float(explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[1])

    # Rank features by contribution
    pos_indices = [idx for idx in np.argsort(shap_vec)[::-1] if shap_vec[idx] > 0]
    neg_indices = [idx for idx in np.argsort(shap_vec) if shap_vec[idx] < 0]

    # Top risk-increasing features
    risk_increasing = []
    adverse_reasons = []
    for idx in pos_indices[:top_n]:
        feat_name = feature_names[idx]
        val = float(shap_vec[idx])
        reason = ADVERSE_ACTION_REASONS.get(
            feat_name,
            f"Adverse credit bureau profile indicator: {feat_name}"
        )
        risk_increasing.append({
            "feature": feat_name,
            "shap_value": round(val, 4),
            "reason": reason,
        })
        if reason not in adverse_reasons:
            adverse_reasons.append(reason)

    # Top risk-reducing features (strengths)
    risk_reducing = []
    for idx in neg_indices[:top_n]:
        feat_name = feature_names[idx]
        val = float(shap_vec[idx])
        risk_reducing.append({
            "feature": feat_name,
            "shap_value": round(val, 4),
        })

    return {
        "base_value": round(base_value, 4),
        "total_shap_sum": round(float(np.sum(shap_vec)), 4),
        "risk_increasing_factors": risk_increasing,
        "risk_reducing_factors": risk_reducing,
        "adverse_action_reasons": adverse_reasons,
        "all_shap_values": dict(zip(feature_names, [round(float(v), 4) for v in shap_vec])),
    }


def predict_batch(
    df: pd.DataFrame,
    model_bundle: Optional[Dict[str, Any]] = None,
    custom_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Score a batch of loan applicants and return an annotated DataFrame."""
    if model_bundle is None:
        model_bundle = load_production_model()

    pipeline = model_bundle["pipeline"]
    opt_t = custom_threshold if custom_threshold is not None else model_bundle.get("optimal_threshold", 0.370)

    df_out = df.copy()
    probs = pipeline.predict_proba(df)[:, 1]
    df_out["default_probability"] = np.round(probs, 4)
    df_out["underwriting_decision"] = np.where(probs >= opt_t, "REJECT", "ACCEPT")

    # Risk tiers
    conditions = [
        probs < 0.15,
        (probs >= 0.15) & (probs < 0.30),
        (probs >= 0.30) & (probs < 0.45),
        probs >= 0.45,
    ]
    tiers = ["Tier 1: Prime", "Tier 2: Near-Prime", "Tier 3: Subprime", "Tier 4: Deep Subprime"]
    df_out["risk_tier"] = np.select(conditions, tiers, default="Unknown")

    return df_out
