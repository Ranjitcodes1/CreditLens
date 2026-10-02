"""
Inference Scoring and Explainability Engine for Loan Default Prediction.
Compliant with Reserve Bank of India (RBI) Fair Practices Code,
CICRA Bureau Reporting Standards, and FCRA/ECOA frameworks.

Provides:
1. Production model loading and pipeline execution.
2. Real-time applicant scoring with native Indian Rupee (INR / ₹) and USD currency support.
3. Expected loss estimation (EL = PD * EAD * LGD).
4. Local Tree SHAP attributions and automated RBI-compliant adverse decision reason codes.
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

# Standard currency exchange conversion factor (1 USD ≈ 83.0 INR)
INR_PER_USD: float = 83.0
MONETARY_FIELDS: List[str] = ["loan_amnt", "annual_inc", "installment", "revol_bal"]

# Regulatory Adverse Decision Reason Codes (RBI Fair Practices Code & CICRA Bureau Standards)
ADVERSE_ACTION_REASONS: Dict[str, str] = {
    "int_rate": "High assigned interest rate reflecting elevated credit assessment risk grade",
    "sub_grade_ord": "Subprime internal risk rating band assigned during underwriting evaluation",
    "grade_ord": "Low credit bureau rating score / elevated risk band",
    "loan_to_income": "Requested loan amount exceeds permissible leverage relative to verified annual income",
    "installment_to_income": "Monthly EMI obligation exceeds permissible fixed obligation-to-income ratio (FOIR)",
    "revol_bal_to_income": "High revolving credit card balance outstanding relative to annual gross earnings",
    "dti": "High Debt-to-Income (DTI) ratio across existing monthly debt obligations",
    "high_dti_flag": "Total monthly non-mortgage debt service burden exceeds underwriting policy limit (>= 20%)",
    "revol_util": "High credit card / revolving line credit limit utilization (near maximum borrowing capacity)",
    "high_utilization_flag": "Revolving credit line utilization exceeds threshold limit (>= 80%)",
    "inq_last_6mths": "Multiple recent credit bureau inquiries reflecting high credit-seeking velocity (CIBIL/bureau report)",
    "velocity_hazard_flag": "Frequent recent loan/credit inquiries observed across bureau records",
    "term": "Extended loan tenure (60-month maturity introduces prolonged credit exposure)",
    "derogatory_flag": "Presence of historical default, legal recovery, or bankruptcy record on bureau file",
    "pub_rec": "Derogatory public civil record or bureau write-off listed on credit history",
    "pub_rec_bankruptcies": "Prior recorded history of insolvency / bankruptcy filing",
    "annual_inc": "Insufficient verified annual gross income to service requested credit facility",
    "log_annual_inc": "Low verified earnings capacity relative to existing debt liabilities",
    "credit_history_years": "Short bureau credit history duration / limited credit seasoning record",
    "delinq_2yrs": "History of past-due payments (30+ days overdue) within the previous 24 months",
    "open_acc": "Inadequate number of established active credit trade lines",
    "total_acc": "Limited historical credit bureau trade line depth",
}


def load_production_model(model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load the serialized champion model bundle from disk."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path.resolve()}")

    bundle = joblib.load(path)
    return bundle


def _convert_applicant_to_model_scale(df: pd.DataFrame, currency: str) -> pd.DataFrame:
    """Normalize monetary fields to training baseline scale (USD) if provided in INR."""
    df_norm = df.copy()
    if currency.upper() == "INR":
        for col in MONETARY_FIELDS:
            if col in df_norm.columns:
                df_norm[col] = df_norm[col].astype(float) / INR_PER_USD
    return df_norm


def predict_loan_applicant(
    applicant_data: Union[Dict[str, Any], pd.DataFrame],
    model_bundle: Optional[Dict[str, Any]] = None,
    custom_threshold: Optional[float] = None,
    lgd: float = 0.60,
    currency: str = "INR",
) -> Dict[str, Any]:
    """Score a loan applicant and generate underwriting decisions under RBI/global credit standards.

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
    currency : str, default='INR'
        Currency of input monetary amounts ('INR' or 'USD').

    Returns
    -------
    dict
        Scoring result containing probability, decision, risk tier, expected loss, and currency symbol.
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

    # Normalize monetary values for model inference
    df_model = _convert_applicant_to_model_scale(df_app, currency=currency)

    # Predict default probability
    p_default = float(pipeline.predict_proba(df_model)[0, 1])

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

    # Calculate Expected Loss (EL = PD * EAD * LGD) in applicant's native currency
    loan_amount = float(df_app["loan_amnt"].values[0]) if "loan_amnt" in df_app.columns else 500000.0
    expected_loss = p_default * loan_amount * lgd
    currency_symbol = "Rs." if currency.upper() == "INR" else "$"

    return {
        "default_probability": round(p_default, 4),
        "default_probability_pct": round(p_default * 100, 2),
        "decision": decision,
        "decision_threshold": round(decision_threshold, 3),
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "loan_amount": loan_amount,
        "expected_loss": round(expected_loss, 2),
        "currency": currency.upper(),
        "currency_symbol": currency_symbol,
        "applicant_input": df_app.iloc[0].to_dict(),
    }


def explain_applicant_shap(
    applicant_data: Union[Dict[str, Any], pd.DataFrame],
    model_bundle: Optional[Dict[str, Any]] = None,
    top_n: int = 5,
    currency: str = "INR",
) -> Dict[str, Any]:
    """Compute local SHAP feature attributions and generate RBI-compliant adverse decision reason codes.

    Parameters
    ----------
    applicant_data : dict or pd.DataFrame
        Single applicant record.
    model_bundle : dict, optional
        Loaded model bundle.
    top_n : int, default=5
        Number of top risk-increasing factors to return.
    currency : str, default='INR'
        Currency of input monetary attributes ('INR' or 'USD').

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

    # Scale monetary inputs if in INR
    df_model = _convert_applicant_to_model_scale(df_app, currency=currency)

    # Deconstruct pipeline steps to access preprocessed vector and classifier
    fe_step = pipeline.named_steps["domain_features"]
    prep_step = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform applicant record through preprocessing
    X_fe = fe_step.transform(df_model)
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

    base_val = explainer.expected_value
    base_value = float(base_val if not isinstance(base_val, (list, np.ndarray)) else base_val[1])

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
    currency: str = "USD",
) -> pd.DataFrame:
    """Score a batch of loan applicants and return an annotated DataFrame."""
    if model_bundle is None:
        model_bundle = load_production_model()

    pipeline = model_bundle["pipeline"]
    opt_t = custom_threshold if custom_threshold is not None else model_bundle.get("optimal_threshold", 0.370)

    df_model = _convert_applicant_to_model_scale(df, currency=currency)
    probs = pipeline.predict_proba(df_model)[:, 1]

    df_out = df.copy()
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
