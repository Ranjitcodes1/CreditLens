"""
Automated Test Suite for Loan Default Prediction Pipeline.
Uses Python standard library unittest.
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import DomainFeatureExtractor, build_feature_pipeline
from src.predict import (
    load_production_model,
    predict_loan_applicant,
    explain_applicant_shap,
    predict_batch,
    DEFAULT_MODEL_PATH,
)


class TestLoanPredictionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and sample applicant data."""
        cls.data_path = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
        cls.assertTrue(cls.data_path.exists(), f"Processed data missing at: {cls.data_path}")

        cls.df = pd.read_parquet(cls.data_path)
        cls.sample_applicant = cls.df.iloc[[0]].drop(columns=["target"])

        # Load model bundle
        cls.assertTrue(DEFAULT_MODEL_PATH.exists(), f"Model artifact missing at: {DEFAULT_MODEL_PATH}")
        cls.bundle = load_production_model()

    def test_01_processed_data_health(self):
        """Verify cleaned dataset has correct structure and zero NaNs."""
        self.assertGreater(len(self.df), 30000)
        self.assertEqual(self.df.isnull().sum().sum(), 0)
        self.assertIn("target", self.df.columns)
        self.assertIn("int_rate", self.df.columns)
        self.assertIn("annual_inc", self.df.columns)

    def test_02_feature_extractor(self):
        """Verify DomainFeatureExtractor generates valid financial metrics."""
        extractor = DomainFeatureExtractor()
        transformed = extractor.fit_transform(self.sample_applicant)

        expected_features = [
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
        for feat in expected_features:
            self.assertIn(feat, transformed.columns)
            self.assertFalse(transformed[feat].isnull().any())

    def test_03_model_bundle_integrity(self):
        """Verify serialized model bundle contains required pipeline and metadata."""
        self.assertIn("pipeline", self.bundle)
        self.assertIn("optimal_threshold", self.bundle)
        self.assertIn("test_metrics", self.bundle)
        self.assertIn("feature_names", self.bundle)

        self.assertGreater(self.bundle["optimal_threshold"], 0.10)
        self.assertLess(self.bundle["optimal_threshold"], 0.60)
        self.assertGreater(self.bundle["test_metrics"]["roc_auc"], 0.65)

    def test_04_single_applicant_prediction(self):
        """Verify single applicant scoring generates probability, decision, and risk tier."""
        res = predict_loan_applicant(self.sample_applicant, model_bundle=self.bundle)

        self.assertIn(res["decision"], ["ACCEPT", "REJECT"])
        self.assertGreaterEqual(res["default_probability"], 0.0)
        self.assertLessEqual(res["default_probability"], 1.0)
        self.assertIn("Tier", res["risk_tier"])
        self.assertGreaterEqual(res["expected_loss"], 0.0)

    def test_05_shap_explainability_and_adverse_reasons(self):
        """Verify SHAP attributions and adverse action reason code generation."""
        explanation = explain_applicant_shap(self.sample_applicant, model_bundle=self.bundle, top_n=4)

        self.assertIn("risk_increasing_factors", explanation)
        self.assertIn("adverse_action_reasons", explanation)
        self.assertGreater(len(explanation["risk_increasing_factors"]), 0)
        self.assertGreater(len(explanation["adverse_action_reasons"]), 0)

        # Check reason text formatting
        for reason in explanation["adverse_action_reasons"]:
            self.assertIsInstance(reason, str)
            self.assertGreater(len(reason), 10)

    def test_06_batch_prediction(self):
        """Verify batch evaluation annotates DataFrame with decision columns."""
        sample_batch = self.df.head(25).drop(columns=["target"])
        scored_batch = predict_batch(sample_batch, model_bundle=self.bundle)

        self.assertEqual(len(scored_batch), 25)
        self.assertIn("default_probability", scored_batch.columns)
        self.assertIn("underwriting_decision", scored_batch.columns)
        self.assertIn("risk_tier", scored_batch.columns)
        self.assertTrue(set(scored_batch["underwriting_decision"].unique()).issubset({"ACCEPT", "REJECT"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
