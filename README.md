# CreditLens — Institutional Credit Risk & Explainable AI Underwriting Engine

An end-to-end, production-grade machine learning credit underwriting platform compliant with the **Reserve Bank of India (RBI) Digital Lending Guidelines (2022/2023)**, **Fair Practices Code for NBFCs**, and **CICRA 2005 Bureau Reporting Standards**.

CreditLens bridges calibrated risk estimation ($Cost_{\text{FN}} = 10 \times Cost_{\text{FP}}$) with explainable TreeSHAP attributions and features a dual-interface architecture: a **state-of-the-art React 18 + Vite + Tailwind CSS web studio** and an **interactive Streamlit data science portal**.

---

## 🌟 Key Architecture & Features

### 1. Modern Web Application (`frontend/`)
* **Zero-Scroll Glassmorphic Cockpit:** Designed for institutional underwriters with a translucent floating layout operating over a live radar canvas ([SonarGrid](file:///frontend/src/components/ui/sonar-grid.tsx)).
* **Character-Level Cut Reveal:** High-impact dynamic typography powered by Framer Motion ([VerticalCutReveal](file:///frontend/src/components/ui/vertical-cut-reveal.tsx)).
* **Glowing Segmented Duration Switcher:** 5-tier flexible repayment tenors (`12M`, `24M`, `36M`, `48M`, `60M`) with animated glowing pill physics.
* **Instant EMI & Default Probability:** Real-time calculation of monthly repayment obligations (in Indian Rupees ₹), risk-based APR tiers, expected portfolio loss ($EL = PD \times EAD \times LGD$), and default probability.
* **Statutory Adverse Action Notices:** Automated generation of formal Section 6 Adverse Action Letters for rejected applicants, detailing specific bureau and leverage factor codes.

### 2. Machine Learning Pipeline (`src/` & `models/`)
* **Leakage-Free Feature Pipeline:** Custom `ColumnTransformer` with target encoding, missingness indicators, and domain credit ratios (`loan_to_income`, `installment_to_income`, `credit_history_years`).
* **Calibrated LightGBM Classifier:** 5-Fold Stratified Cross-Validation achieving **0.7151 Test ROC-AUC** and **0.3023 Test PR-AUC** on imbalanced loan portfolios (14.59% default rate).
* **Cost-Sensitive Threshold Optimization:** Asymmetric loss optimization setting $\tau^* = 0.360$ to minimize aggregate charge-off dollars ($>\$2.85\text{M}$ saved on validation portfolios).
* **Local TreeSHAP Explainability:** Translates feature attribution vectors into protective factors vs. risk drivers in real time.

---

## 📁 Repository Structure

```text
creditlens/
├── frontend/                   # Modern React 18 Web Application
│   ├── src/
│   │   ├── components/         # BrandHero.tsx, LoanPredictorStudio.tsx
│   │   │   └── ui/             # OriginButton, SonarGrid, VerticalCutReveal, etc.
│   │   ├── lib/                # Tailwind & utility helpers
│   │   ├── App.tsx             # State-driven zero-scroll view controller
│   │   └── main.tsx            # Application entrypoint
│   ├── package.json            # React, Vite, Tailwind, Framer Motion dependencies
│   ├── tailwind.config.js      # Custom theme, tokens, and animations
│   ├── tsconfig.json           # TypeScript configuration with @/* path aliases
│   └── vite.config.ts          # Vite bundler configuration
├── src/                        # Production Machine Learning Source Code
│   ├── data_prep.py            # Data loading, target definition, leakage filtering
│   ├── features.py             # Sklearn ColumnTransformer & domain transformers
│   ├── train.py                # Stratified CV, training routines, Optuna tuning
│   ├── predict.py              # Inference engine, INR currency conversion & SHAP
│   └── generate_eda_figures.py # Diagnostic plotting scripts
├── models/                     # Serialized Model Artifacts
│   └── loan_default_model.joblib # Trained production LightGBM pipeline bundle
├── app/                        # Python Analytics Portal
│   └── streamlit_app.py        # Streamlit underwriting & portfolio dashboard
├── notebooks/                  # Guided Exploratory Jupyter Notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_explainability.ipynb
├── requirements.txt            # Python dependencies (LightGBM, SHAP, Optuna, etc.)
└── README.md
```

---

## 🚀 Quick Start Guide

### Option A: Launch the CreditLens Web Studio (React + Vite)

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the local development server (runs on http://localhost:3000)
npm run dev

# 4. (Optional) Compile production bundle
npm run build
```

### Option B: Launch the Streamlit Analytics Explorer (Python)

```bash
# 1. Create and activate a Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # On Windows PowerShell
# source .venv/bin/activate   # On Linux/macOS

# 2. Install Python requirements
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app/streamlit_app.py
```

---

## 🏆 Model Benchmarks & Validation Results

Evaluated across **5-Fold Stratified Cross-Validation** and a **20% held-out test set** (38,577 closed loans):

| Architecture | 5-Fold ROC-AUC | 5-Fold PR-AUC | Brier Score | Recall@0.50 | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Champion)** | **0.7087 ± 0.0107** | **0.2921 ± 0.0140** | **0.2040** | **61.8%** | **0.7151** | **0.3023** |
| **Logistic Regression** | 0.7100 ± 0.0080 | 0.2935 ± 0.0108 | 0.2152 | 63.4% | 0.7124 | 0.2985 |
| **XGBoost** | 0.7030 ± 0.0110 | 0.2857 ± 0.0129 | 0.2013 | 58.3% | 0.7072 | 0.2918 |
| **Random Forest** | 0.6966 ± 0.0109 | 0.2770 ± 0.0078 | 0.1898 | 51.5% | 0.7001 | 0.2842 |

### 💰 Cost-Sensitive Threshold Calibration

Underwriting carries an asymmetric financial penalty where approving a defaulting borrower is **10x more costly** than declining a solvent one:
* **Cost of False Negative ($C_{\text{FN}}$)**: $\approx \$5,000$ (unrecovered principal upon charge-off)
* **Cost of False Positive ($C_{\text{FP}}$)**: $\approx \$500$ (lost net interest margin and customer acquisition cost)

```text
Decision Policy Comparison on Held-Out Test Set (7,716 loans):
┌──────────────────────────────┬───────────┬────────────────┬────────────────────────┐
│ Underwriting Strategy        │ Threshold │ Aggregate Loss │ Dollar Savings         │
├──────────────────────────────┼───────────┼────────────────┼────────────────────────┤
│ Unhedged (Approve All)       │    1.00   │  $5,625,000.00 │ Baseline               │
│ Standard Cutoff (Naive 0.50) │    0.50   │  $3,181,000.00 │ $2,444,000.00 (43.4%)  │
│ Cost-Optimal Policy (Tau*)   │    0.36   │  $2,770,000.00 │ $2,855,000.00 (50.8%)  │
└──────────────────────────────┴───────────┴────────────────┴────────────────────────┘
```
> Shifting from naive 0.50 to the calibrated **0.360** threshold captures **86.2% of all portfolio defaults**, saving **$2.85M** in bad debt write-offs.

---

## ⚖️ Regulatory Compliance & Governance

* **Reserve Bank of India (RBI) Digital Lending Guidelines:** Strictly enforces fair interest rate determination and transparent algorithm explainability.
* **Adverse Action Disclosures (CICRA 2005 & Section 6 FPC):** Automatically translates negative TreeSHAP attribution drivers into specific statutory reasons for declined applicants.
* **Zero Target Leakage:** Rigid segregation of post-origination servicing fields (`recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`, `total_rec_prncp`).

---

## 🛠️ CLI Training & Scoring

```bash
# Retrain pipeline from raw data
python src/train.py

# Run Bayesian hyperparameter tuning with Optuna (25 trials)
python src/train.py --tune --trials 25

# Score an individual applicant via Python CLI
python src/predict.py
```
