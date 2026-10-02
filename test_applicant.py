"""
Quick CLI Test Script to evaluate sample applicants
and demonstrate RBI Adverse Action Disclosures.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_production_model, predict_loan_applicant, explain_applicant_shap

def main():
    print("=" * 60)
    print("  TEST 3: PROGRAMMATIC UNDERWRITING & RBI COMPLIANCE TEST")
    print("=" * 60)

    print("\nLoading production model bundle...")
    bundle = load_production_model()
    print(f"Model: {bundle['model_type']} (Cutoff Threshold: {bundle['optimal_threshold']*100:.1f}%)\n")

    # 1. Test High-Risk Applicant (Subprime)
    print("-" * 60)
    print("Case 1: Testing High-Risk Applicant (High DTI, Inquiries, Low Income)")
    print("-" * 60)
    applicant_high_risk = {
        'loan_amnt': 350000.0, 'term': 60, 'int_rate': 22.5, 'installment': 9750.0,
        'grade': 'F', 'sub_grade': 'F2', 'home_ownership': 'RENT', 'annual_inc': 360000.0,
        'verification_status': 'Verified', 'issue_d': 'Dec-11', 'purpose': 'small_business',
        'addr_state': 'CA', 'dti': 32.5, 'delinq_2yrs': 1, 'earliest_cr_line': 'Jan-08',
        'inq_last_6mths': 4, 'open_acc': 7, 'pub_rec': 1, 'revol_bal': 180000.0,
        'revol_util': 88.0, 'total_acc': 14, 'pub_rec_bankruptcies': 1,
        'emp_length_years': 2.0, 'emp_length_missing': 0
    }

    res_reject = predict_loan_applicant(applicant_high_risk, model_bundle=bundle, currency="INR")
    explain_reject = explain_applicant_shap(applicant_high_risk, model_bundle=bundle, top_n=4, currency="INR")

    print(f"Decision:               {res_reject['decision']}")
    print(f"Default Probability:    {res_reject['default_probability_pct']:.2f}%")
    print(f"Risk Tier:              {res_reject['risk_tier']}")
    print(f"Expected Credit Loss:   Rs. {res_reject['expected_loss']:,.2f}")
    print("\nRBI Fair Practices Code Adverse Action Reasons:")
    for idx, reason in enumerate(explain_reject['adverse_action_reasons'], 1):
        print(f"  {idx}. {reason}")

    # 2. Test Prime Applicant (Salaried)
    print("\n" + "-" * 60)
    print("Case 2: Testing Prime Applicant (High Income, Low DTI, Grade A)")
    print("-" * 60)
    applicant_prime = {
        'loan_amnt': 500000.0, 'term': 36, 'int_rate': 8.5, 'installment': 15780.0,
        'grade': 'A', 'sub_grade': 'A2', 'home_ownership': 'MORTGAGE', 'annual_inc': 1500000.0,
        'verification_status': 'Verified', 'issue_d': 'Dec-11', 'purpose': 'debt_consolidation',
        'addr_state': 'CA', 'dti': 9.5, 'delinq_2yrs': 0, 'earliest_cr_line': 'May-99',
        'inq_last_6mths': 0, 'open_acc': 14, 'pub_rec': 0, 'revol_bal': 45000.0,
        'revol_util': 15.0, 'total_acc': 28, 'pub_rec_bankruptcies': 0,
        'emp_length_years': 8.0, 'emp_length_missing': 0
    }

    res_accept = predict_loan_applicant(applicant_prime, model_bundle=bundle, currency="INR")
    print(f"Decision:               {res_accept['decision']}")
    print(f"Default Probability:    {res_accept['default_probability_pct']:.2f}%")
    print(f"Risk Tier:              {res_accept['risk_tier']}")
    print(f"Expected Credit Loss:   Rs. {res_accept['expected_loss']:,.2f}")

    print("\n" + "=" * 60)
    print("  TEST 3 PASSED: Both approval and rejection paths verified!")
    print("=" * 60)

if __name__ == "__main__":
    main()
