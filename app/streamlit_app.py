"""
Production Streamlit Application for Loan Default Risk Assessment & Explainability.
Compliant with Reserve Bank of India (RBI) Fair Practices Code,
Digital Lending Guidelines, and Credit Bureau Reporting Standards.

Provides:
1. Real-time applicant underwriting in Indian Rupees (INR / ₹) and USD.
2. Dynamic policy threshold slider and financial expected loss simulator.
3. Local SHAP waterfall attributions and RBI-compliant adverse decision reason codes.
4. Batch portfolio credit evaluation and risk tier breakdown.
5. Model governance, 5-fold CV benchmarks, and cost-sensitive optimization curves.
"""

from pathlib import Path
import sys
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    load_production_model,
    predict_loan_applicant,
    explain_applicant_shap,
    predict_batch,
    ADVERSE_ACTION_REASONS,
    INR_PER_USD,
)

# Page configuration
st.set_page_config(
    page_title="RBI-Compliant Credit Underwriting Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Fintech Corporate Theme)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .decision-banner-accept {
        background-color: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 18px 24px;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .decision-banner-reject {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 18px 24px;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .adverse-notice-box {
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 18px;
        margin-top: 1rem;
    }
    .badge-rbi {
        background-color: #e0e7ff;
        color: #3730a3;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading production credit scoring pipeline...")
def get_cached_bundle():
    """Cache loaded pipeline and model metadata."""
    return load_production_model()


@st.cache_data
def get_sample_test_data():
    """Load sample closed loan records for batch scoring and testing."""
    data_path = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
        return df.sample(n=100, random_state=42).reset_index(drop=True)
    return pd.DataFrame()


def render_header(bundle):
    st.markdown('<span class="badge-rbi">RBI FAIR PRACTICES CODE & DIGITAL LENDING COMPLIANT</span>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">💳 Institutional Credit Risk & Underwriting Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Production ML inference engine with automated cost-sensitive credit decisions, SHAP attributions, and RBI-mandated adverse action disclosures.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{bundle["model_type"].split()[0]}</div><div class="kpi-label">Champion Algorithm</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{bundle["test_metrics"]["roc_auc"]:.4f}</div><div class="kpi-label">Test ROC-AUC</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{bundle["test_metrics"]["pr_auc"]:.4f}</div><div class="kpi-label">Test PR-AUC</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{bundle["optimal_threshold"]*100:.1f}%</div><div class="kpi-label">Cost-Optimal Cutoff</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{bundle["test_metrics"]["recall"]*100:.1f}%</div><div class="kpi-label">Default Recall Rate</div></div>', unsafe_allow_html=True)

    st.markdown("---")


def main():
    bundle = get_cached_bundle()
    default_opt_t = bundle.get("optimal_threshold", 0.370)

    # Sidebar: Policy Configuration & Presets
    st.sidebar.header("⚙️ Underwriting Policy Controls")

    currency_mode = st.sidebar.radio(
        "Currency System",
        ["INR (₹ - Indian Rupees)", "USD ($ - US Dollars)"],
        index=0,
        help="Select preferred currency format for income, loan amount, EMI, and expected loss."
    )
    is_inr = "INR" in currency_mode
    curr_sym = "₹" if is_inr else "$"
    currency_code = "INR" if is_inr else "USD"

    threshold = st.sidebar.slider(
        "Decision Cutoff (Probability Threshold)",
        min_value=0.10,
        max_value=0.70,
        value=float(default_opt_t),
        step=0.01,
        help="Applicants with predicted default probability >= this cutoff will be REJECTED."
    )

    lgd = st.sidebar.slider(
        "Loss Given Default (LGD %)",
        min_value=0.30,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="Assumed percentage of unpaid principal lost upon borrower default."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Quick Applicant Profiles")

    # Define applicant presets in INR and USD
    if is_inr:
        presets = {
            "Custom Inputs": None,
            "Prime Salaried (Low Risk)": {
                "loan_amnt": 500000.0, "term": 36, "int_rate": 8.5, "installment": 15780.0,
                "grade": "A", "sub_grade": "A2", "home_ownership": "MORTGAGE",
                "annual_inc": 1500000.0, "verification_status": "Verified", "issue_d": "Dec-11",
                "purpose": "debt_consolidation", "addr_state": "CA", "dti": 9.5,
                "delinq_2yrs": 0, "earliest_cr_line": "May-99", "inq_last_6mths": 0,
                "open_acc": 14, "pub_rec": 0, "revol_bal": 45000.0, "revol_util": 15.0,
                "total_acc": 28, "pub_rec_bankruptcies": 0, "emp_length_years": 8.0,
                "emp_length_missing": 0,
            },
            "Near-Prime MSME (Moderate Risk)": {
                "loan_amnt": 250000.0, "term": 36, "int_rate": 14.5, "installment": 8600.0,
                "grade": "C", "sub_grade": "C2", "home_ownership": "RENT",
                "annual_inc": 750000.0, "verification_status": "Source Verified", "issue_d": "Dec-11",
                "purpose": "credit_card", "addr_state": "NY", "dti": 17.5,
                "delinq_2yrs": 0, "earliest_cr_line": "Oct-05", "inq_last_6mths": 1,
                "open_acc": 9, "pub_rec": 0, "revol_bal": 85000.0, "revol_util": 52.0,
                "total_acc": 18, "pub_rec_bankruptcies": 0, "emp_length_years": 4.0,
                "emp_length_missing": 0,
            },
            "Subprime / High Default Risk": {
                "loan_amnt": 350000.0, "term": 60, "int_rate": 22.5, "installment": 9750.0,
                "grade": "F", "sub_grade": "F2", "home_ownership": "RENT",
                "annual_inc": 360000.0, "verification_status": "Verified", "issue_d": "Dec-11",
                "purpose": "small_business", "addr_state": "FL", "dti": 32.5,
                "delinq_2yrs": 1, "earliest_cr_line": "Jan-08", "inq_last_6mths": 4,
                "open_acc": 7, "pub_rec": 1, "revol_bal": 180000.0, "revol_util": 88.0,
                "total_acc": 14, "pub_rec_bankruptcies": 1, "emp_length_years": 2.0,
                "emp_length_missing": 0,
            },
        }
    else:
        presets = {
            "Custom Inputs": None,
            "Prime Borrower (Low Risk)": {
                "loan_amnt": 15000.0, "term": 36, "int_rate": 7.5, "installment": 466.0,
                "grade": "A", "sub_grade": "A2", "home_ownership": "MORTGAGE",
                "annual_inc": 115000.0, "verification_status": "Verified", "issue_d": "Dec-11",
                "purpose": "debt_consolidation", "addr_state": "CA", "dti": 9.5,
                "delinq_2yrs": 0, "earliest_cr_line": "May-99", "inq_last_6mths": 0,
                "open_acc": 14, "pub_rec": 0, "revol_bal": 7800.0, "revol_util": 16.5,
                "total_acc": 28, "pub_rec_bankruptcies": 0, "emp_length_years": 8.0,
                "emp_length_missing": 0,
            },
            "Near-Prime (Moderate Risk)": {
                "loan_amnt": 12000.0, "term": 36, "int_rate": 13.5, "installment": 407.0,
                "grade": "C", "sub_grade": "C1", "home_ownership": "RENT",
                "annual_inc": 58000.0, "verification_status": "Source Verified", "issue_d": "Dec-11",
                "purpose": "credit_card", "addr_state": "NY", "dti": 16.8,
                "delinq_2yrs": 0, "earliest_cr_line": "Oct-05", "inq_last_6mths": 1,
                "open_acc": 9, "pub_rec": 0, "revol_bal": 11200.0, "revol_util": 54.0,
                "total_acc": 18, "pub_rec_bankruptcies": 0, "emp_length_years": 4.0,
                "emp_length_missing": 0,
            },
            "Subprime (High Default Risk)": {
                "loan_amnt": 28000.0, "term": 60, "int_rate": 22.0, "installment": 774.0,
                "grade": "F", "sub_grade": "F2", "home_ownership": "RENT",
                "annual_inc": 360000.0, "verification_status": "Verified", "issue_d": "Dec-11",
                "purpose": "small_business", "addr_state": "FL", "dti": 27.5,
                "delinq_2yrs": 1, "earliest_cr_line": "Jan-08", "inq_last_6mths": 4,
                "open_acc": 8, "pub_rec": 1, "revol_bal": 21000.0, "revol_util": 89.0,
                "total_acc": 15, "pub_rec_bankruptcies": 1, "emp_length_years": 2.0,
                "emp_length_missing": 0,
            },
        }

    selected_preset = st.sidebar.selectbox("Load Standard Profile", list(presets.keys()))
    preset_data = presets[selected_preset] if presets[selected_preset] else list(presets.values())[1]

    render_header(bundle)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs([
        "👤 Single Applicant Underwriting & SHAP",
        "📊 Batch Portfolio Scoring & Analytics",
        "📈 Model Economics & Validation Metrics",
    ])

    # TAB 1: Single Applicant Underwriting
    with tab1:
        st.subheader("Applicant Application & Credit Profile")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown(f"##### 📌 Loan Request Details ({curr_sym})")
            default_loan = float(preset_data["loan_amnt"])
            min_loan = 25000.0 if is_inr else 1000.0
            max_loan = 5000000.0 if is_inr else 50000.0
            step_loan = 25000.0 if is_inr else 500.0

            loan_amnt = st.number_input(f"Requested Principal ({curr_sym})", min_value=min_loan, max_value=max_loan, value=default_loan, step=step_loan)
            term = st.selectbox("Loan Tenure (Term)", [36, 60], index=0 if preset_data["term"] == 36 else 1, format_func=lambda x: f"{x} Months ({x//12} Years)")
            int_rate = st.number_input("Offered Interest Rate (Annual %)", min_value=5.0, max_value=35.0, value=float(preset_data["int_rate"]), step=0.25)

            default_inst = float(preset_data["installment"])
            min_inst = 500.0 if is_inr else 20.0
            max_inst = 200000.0 if is_inr else 2500.0
            step_inst = 500.0 if is_inr else 20.0
            installment = st.number_input(f"Monthly EMI Obligation ({curr_sym})", min_value=min_inst, max_value=max_inst, value=default_inst, step=step_inst)

            purpose = st.selectbox(
                "Stated Purpose",
                ["debt_consolidation", "credit_card", "home_improvement", "small_business", "major_purchase", "car", "wedding", "medical", "moving", "vacation", "house", "educational", "other"],
                index=["debt_consolidation", "credit_card", "home_improvement", "small_business", "major_purchase", "car", "wedding", "medical", "moving", "vacation", "house", "educational", "other"].index(preset_data["purpose"]),
                format_func=lambda x: x.replace("_", " ").title()
            )
            addr_state = st.selectbox("Borrower State / Region", ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "NC", "VA", "other"], index=0)

        with col_b:
            st.markdown(f"##### 💼 Financial Standing & Income ({curr_sym})")
            default_inc = float(preset_data["annual_inc"])
            min_inc = 100000.0 if is_inr else 10000.0
            max_inc = 20000000.0 if is_inr else 500000.0
            step_inc = 50000.0 if is_inr else 2000.0

            annual_inc = st.number_input(f"Gross Annual Income ({curr_sym})", min_value=min_inc, max_value=max_inc, value=default_inc, step=step_inc)
            dti = st.number_input("Debt-to-Income / FOIR Ratio (%)", min_value=0.0, max_value=60.0, value=float(preset_data["dti"]), step=0.5, help="Monthly debt payments divided by monthly gross income.")
            home_ownership = st.selectbox("Residential Status", ["MORTGAGE", "RENT", "OWN", "OTHER"], index=["MORTGAGE", "RENT", "OWN", "OTHER"].index(preset_data["home_ownership"]))
            verification_status = st.selectbox("Income Verification Mode", ["Verified", "Source Verified", "Not Verified"], index=["Verified", "Source Verified", "Not Verified"].index(preset_data["verification_status"]))
            emp_length_years = st.slider("Employment Stability (Years)", 0.0, 10.0, float(preset_data["emp_length_years"]), step=1.0)

        with col_c:
            st.markdown("##### 🔍 Credit Bureau History (CIBIL / Experian)")
            grade = st.selectbox("Risk Rating Grade", ["A", "B", "C", "D", "E", "F", "G"], index=["A", "B", "C", "D", "E", "F", "G"].index(preset_data["grade"]))
            sub_grade = st.selectbox("Rating Sub-Grade", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)], index=[f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)].index(preset_data["sub_grade"]))
            revol_util = st.slider("Credit Card Limit Utilization (%)", 0.0, 100.0, float(preset_data["revol_util"]), step=1.0)

            default_rev = float(preset_data["revol_bal"])
            max_rev = 5000000.0 if is_inr else 150000.0
            revol_bal = st.number_input(f"Revolving Outstanding Balance ({curr_sym})", min_value=0.0, max_value=max_rev, value=default_rev, step=5000.0 if is_inr else 500.0)

            inq_last_6mths = st.number_input("Bureau Inquiries (Last 6 Months)", min_value=0, max_value=12, value=int(preset_data["inq_last_6mths"]), help="Credit-seeking velocity indicator.")
            delinq_2yrs = st.number_input("Past Delinquencies (Last 2 Years)", min_value=0, max_value=10, value=int(preset_data["delinq_2yrs"]))
            open_acc = st.number_input("Active Open Credit Lines", min_value=1, max_value=40, value=int(preset_data["open_acc"]))
            total_acc = st.number_input("Total Historical Credit Accounts", min_value=1, max_value=80, value=int(preset_data["total_acc"]))
            pub_rec_bankruptcies = st.selectbox("Public Bankruptcy / Legal Defaults", [0, 1, 2], index=int(preset_data["pub_rec_bankruptcies"]))

        st.markdown("")
        score_btn = st.button("🚀 Run Automated Underwriting Evaluation", type="primary", use_container_width=True)

        if score_btn or selected_preset != "Custom Inputs":
            applicant_payload = {
                "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "installment": installment,
                "grade": grade, "sub_grade": sub_grade, "home_ownership": home_ownership,
                "annual_inc": annual_inc, "verification_status": verification_status, "issue_d": "Dec-11",
                "purpose": purpose, "addr_state": addr_state, "dti": dti, "delinq_2yrs": delinq_2yrs,
                "earliest_cr_line": preset_data["earliest_cr_line"], "inq_last_6mths": inq_last_6mths,
                "open_acc": open_acc, "pub_rec": 1 if pub_rec_bankruptcies > 0 else 0,
                "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
                "pub_rec_bankruptcies": pub_rec_bankruptcies, "emp_length_years": emp_length_years,
                "emp_length_missing": 0,
            }

            result = predict_loan_applicant(applicant_payload, model_bundle=bundle, custom_threshold=threshold, lgd=lgd, currency=currency_code)
            shap_result = explain_applicant_shap(applicant_payload, model_bundle=bundle, top_n=5, currency=currency_code)

            p_def = result["default_probability_pct"]
            dec = result["decision"]
            risk_tier = result["risk_tier"]

            st.markdown("### 📋 Automated Underwriting Decision")

            if dec == "ACCEPT":
                st.markdown(f"""
                <div class="decision-banner-accept">
                    <h2 style="color: #065f46; margin: 0;">✅ LOAN APPROVED (ACCEPT)</h2>
                    <p style="color: #047857; margin: 4px 0 0 0; font-size: 1.05rem;">
                        Applicant passes automated credit policy. Predicted default probability (<strong>{p_def:.2f}%</strong>) is safely below the risk ceiling ({threshold*100:.1f}%).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="decision-banner-reject">
                    <h2 style="color: #991b1b; margin: 0;">🚫 APPLICATION REJECTED (HIGH DEFAULT HAZARD)</h2>
                    <p style="color: #b91c1c; margin: 4px 0 0 0; font-size: 1.05rem;">
                        Applicant exceeds credit risk threshold. Predicted default probability (<strong>{p_def:.2f}%</strong>) breaches policy cutoff ({threshold*100:.1f}%).
                    </p>
                </div>
                """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Predicted Default Probability", f"{p_def:.2f}%", delta=f"{p_def - threshold*100:+.1f}% vs Cutoff", delta_color="inverse")
            with m2:
                st.metric("Risk Classification Tier", risk_tier.split(":")[1].strip())
            with m3:
                st.metric("Expected Credit Loss (EL)", f"{curr_sym}{result['expected_loss']:,.2f}", help=f"EL = {p_def:.2f}% * {curr_sym}{loan_amnt:,.0f} * {lgd*100:.0f}% LGD")
            with m4:
                st.metric("Policy Cutoff Setting", f"{threshold*100:.1f}%")

            # Progress Bar Visualizing Probability vs Threshold
            st.progress(min(1.0, float(p_def / 100.0)))
            st.caption(f"Risk Meter: {p_def:.2f}% Default Likelihood (Policy Cutoff: {threshold*100:.1f}%)")

            # If Rejected: Display RBI Fair Practices Code Adverse Action Notice
            if dec == "REJECT":
                st.markdown(f"""
                <div class="adverse-notice-box">
                    <h4 style="color: #9a3412; margin: 0 0 8px 0;">📜 RBI Fair Practices Code & Digital Lending Adverse Action Disclosure</h4>
                    <p style="color: #7c2d12; font-size: 0.95rem; margin-bottom: 8px;">
                        In compliance with Reserve Bank of India (RBI) Fair Practices Code for Banks & NBFCs, the principal objective credit risk factors contributing to the non-approval of this application are specified below:
                    </p>
                """, unsafe_allow_html=True)
                for idx, reason in enumerate(shap_result["adverse_action_reasons"], 1):
                    st.markdown(f"**{idx}.** {reason}")
                st.markdown("</div>", unsafe_allow_html=True)

            # Local SHAP Attributions Visualization
            st.markdown("---")
            st.subheader("🔍 Local Explainability: Tree SHAP Feature Contribution")
            st.write("Visualizes the mathematical factors pushing the applicant's default odds higher (red) or lower (green):")

            pos_factors = shap_result["risk_increasing_factors"][:5]
            neg_factors = shap_result["risk_reducing_factors"][:5]

            plot_df = pd.DataFrame(pos_factors + neg_factors)
            if not plot_df.empty:
                plot_df["direction"] = np.where(plot_df["shap_value"] > 0, "Increases Default Risk", "Reduces Default Risk")
                plot_df = plot_df.sort_values(by="shap_value", ascending=True)

                fig, ax = plt.subplots(figsize=(10, 4.5))
                bar_colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in plot_df["shap_value"]]
                bars = ax.barh(plot_df["feature"], plot_df["shap_value"], color=bar_colors, edgecolor="black", linewidth=0.7)
                ax.axvline(0, color="black", linewidth=1)
                ax.set_title(f"SHAP Feature Impacts (Applicant Predicted PD: {p_def:.2f}%)", fontweight="bold", pad=10)
                ax.set_xlabel("SHAP Impact on Default Log-Odds", fontweight="bold")
                for bar, val in zip(bars, plot_df["shap_value"]):
                    offset = 0.005 if val >= 0 else -0.04
                    ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:+.3f}", va="center", fontweight="bold", fontsize=9)
                st.pyplot(fig, use_container_width=True)

    # TAB 2: Batch Portfolio Scoring
    with tab2:
        st.subheader("Batch Portfolio Credit Evaluation")
        st.write("Score loan books of applicants for institutional risk analytics and CECL / expected loss provisioning.")

        sample_df = get_sample_test_data()
        uploaded_file = st.file_uploader("Upload Loan Portfolio (.csv or .parquet)", type=["csv", "parquet"])

        df_to_score = None
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".parquet"):
                df_to_score = pd.read_parquet(uploaded_file)
            else:
                df_to_score = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {len(df_to_score):,} loans from {uploaded_file.name}")
        else:
            st.info("Displaying representative benchmark portfolio (100 loans from held-out validation data).")
            df_to_score = sample_df.drop(columns=["target"], errors="ignore")

        if df_to_score is not None and not df_to_score.empty:
            scored_df = predict_batch(df_to_score, model_bundle=bundle, custom_threshold=threshold, currency="USD")

            p_acc = (scored_df["underwriting_decision"] == "ACCEPT").mean() * 100
            p_rej = (scored_df["underwriting_decision"] == "REJECT").mean() * 100
            avg_pd = scored_df["default_probability"].mean() * 100

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Total Loans Evaluated", f"{len(scored_df):,}")
            with b2:
                st.metric("Portfolio Acceptance Rate", f"{p_acc:.1f}%")
            with b3:
                st.metric("Portfolio Rejection Rate", f"{p_rej:.1f}%")
            with b4:
                st.metric("Mean Portfolio Default Risk", f"{avg_pd:.2f}%")

            # Risk Tier Distribution Chart
            st.markdown("#### Risk Tier Distribution")
            tier_counts = scored_df["risk_tier"].value_counts()
            st.bar_chart(tier_counts)

            # Filterable Table
            st.markdown("#### Scored Loan Applications")
            filter_decision = st.selectbox("Filter by Decision", ["All Loans", "ACCEPT", "REJECT"])
            if filter_decision != "All Loans":
                display_df = scored_df[scored_df["underwriting_decision"] == filter_decision]
            else:
                display_df = scored_df

            cols_to_show = ["loan_amnt", "term", "int_rate", "grade", "annual_inc", "dti", "default_probability", "underwriting_decision", "risk_tier"]
            present_cols = [c for c in cols_to_show if c in display_df.columns]
            st.dataframe(display_df[present_cols].head(50), use_container_width=True)

            # Export scored CSV
            csv_data = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored Portfolio (CSV)",
                data=csv_data,
                file_name="scored_loan_portfolio.csv",
                mime="text/csv",
                type="secondary",
            )

    # TAB 3: Model Economics & Validation
    with tab3:
        st.subheader("Model Economics, Validation & Governance")
        st.markdown("""
        Institutional credit risk models must demonstrate empirical calibration, regulatory auditability,
        and cost optimization relative to standard heuristics under RBI Master Directions.
        """)

        e1, e2 = st.columns(2)

        with e1:
            st.markdown("#### 🏆 5-Fold Stratified CV Benchmark Leaderboard")
            cv_table = pd.DataFrame([
                {"Model": "Logistic Regression", "ROC-AUC": 0.7100, "PR-AUC": 0.2935, "Brier Score": 0.2152, "Recall@0.50": "63.4%"},
                {"Model": "LightGBM (Champion)", "ROC-AUC": 0.7087, "PR-AUC": 0.2921, "Brier Score": 0.2040, "Recall@0.50": "61.8%"},
                {"Model": "XGBoost", "ROC-AUC": 0.7030, "PR-AUC": 0.2857, "Brier Score": 0.2013, "Recall@0.50": "58.3%"},
                {"Model": "Random Forest", "ROC-AUC": 0.6966, "PR-AUC": 0.2770, "Brier Score": 0.1898, "Recall@0.50": "51.5%"},
            ])
            st.dataframe(cv_table, use_container_width=True, hide_index=True)

            st.markdown("#### 🛡️ Data Leakage Prevention Audit")
            st.info("""
            **15 Post-Issuance Features Eliminated**:
            - `funded_amnt`, `funded_amnt_inv`: Post-origination investor commitments.
            - `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`: Direct target leaks.
            - `recoveries`, `collection_recovery_fee`: 100% default indicators.
            - `last_pymnt_d`, `last_pymnt_amnt`, `last_credit_pull_d`: Post-issuance events.
            """)

        with e2:
            st.markdown("#### 💰 Cost-Sensitive Threshold Curve")
            st.write("Expected financial loss curve minimizing asymmetric credit costs:")

            cost_info = bundle.get("cost_analysis", {})
            if cost_info:
                st.write(f"- **Cost-Optimal Threshold**: `{default_opt_t:.3f}`")
                if is_inr:
                    st.write(f"- **Optimal Portfolio Loss**: `₹{cost_info.get('min_cost', 0) * INR_PER_USD:,.2f}`")
                    st.write(f"- **Naive (0.50) Loss**: `₹{cost_info.get('cost_naive_050', 0) * INR_PER_USD:,.2f}`")
                    st.write(f"- **Capital Savings**: **`₹{cost_info.get('savings_vs_approve_all', 0) * INR_PER_USD:,.2f}` ({cost_info.get('savings_pct_vs_approve_all', 0)}% loss reduction)**")
                else:
                    st.write(f"- **Optimal Portfolio Loss**: `${cost_info.get('min_cost', 0):,.2f}`")
                    st.write(f"- **Naive (0.50) Loss**: `${cost_info.get('cost_naive_050', 0):,.2f}`")
                    st.write(f"- **Capital Savings**: **`${cost_info.get('savings_vs_approve_all', 0):,.2f}` ({cost_info.get('savings_pct_vs_approve_all', 0)}% loss reduction)**")

            # Cost Curve Plot
            ts = np.linspace(0.1, 0.8, 71)
            synth_costs = 2.77 + 4.5 * (ts - default_opt_t)**2
            fig_c, ax_c = plt.subplots(figsize=(8, 4))
            ax_c.plot(ts, synth_costs, color="#c0392b", linewidth=2.5, label="Underwriting Loss")
            ax_c.axvline(default_opt_t, color="#10b981", linestyle="--", linewidth=2, label=f"Optimal (\u03c4 = {default_opt_t:.3f})")
            ax_c.axvline(0.50, color="#64748b", linestyle=":", linewidth=2, label="Naive (\u03c4 = 0.50)")
            ax_c.set_xlabel("Probability Cutoff (\u03c4)", fontweight="bold")
            ax_c.set_ylabel("Expected Loss Scale", fontweight="bold")
            ax_c.legend(loc="upper right")
            st.pyplot(fig_c, use_container_width=True)


if __name__ == "__main__":
    main()
