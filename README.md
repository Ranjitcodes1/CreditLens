# Loan Default Prediction Engine

An end-to-end production-grade machine learning project predicting the likelihood of loan default using Lending Club loan data. Built with strict data leakage prevention, stratified cross-validation, cost-sensitive threshold optimization, SHAP interpretability, and an interactive Streamlit underwriting application.

## 📁 Repository Structure

```text
loan-default-prediction/
├── data/
│   ├── raw/                # Original untouched datasets (git-ignored)
│   └── processed/          # Cleaned, leakage-free feature data (git-ignored)
├── notebooks/              # Guided step-by-step exploratory notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_explainability.ipynb
├── src/                    # Modular, production-ready Python source code
│   ├── __init__.py
│   ├── data_prep.py        # Loading, target definition, leakage filtering
│   ├── features.py         # Sklearn ColumnTransformer & domain transformers
│   ├── train.py            # Stratified CV, training routines, Optuna tuning
│   └── predict.py          # Inference scoring & SHAP feature attribution
├── models/                 # Serialized production pipeline artifacts (.joblib)
├── app/
│   └── streamlit_app.py    # Interactive underwriting decision dashboard
├── reports/
│   └── figures/            # High-resolution visual figures (EDA, SHAP, ROC/PR)
├── requirements.txt        # Pinned project dependencies
├── .gitignore              # Git exclusions for models, raw data, checkpoints
└── README.md
```

## 🛠️ Tech Stack
- **Language**: Python 3.11
- **Data & Processing**: Pandas, NumPy, Scipy
- **Visualization**: Matplotlib, Seaborn
- **Modeling**: scikit-learn (`Pipeline`, `ColumnTransformer`), imbalanced-learn, XGBoost, LightGBM
- **Optimization & Explainability**: Optuna, SHAP, Joblib
- **Deployment**: Streamlit, Git/GitHub

## 🚀 Environment Setup

```bash
# 1. Create a virtual environment using Python 3.11
python -m venv .venv

# 2. Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 🏆 Model Benchmarks & Validation Results

Evaluated across **5-fold Stratified Cross-Validation** and a **20% held-out test set** (38,577 total closed loans, 14.59% baseline default rate):

| Architecture | 5-Fold ROC-AUC | 5-Fold PR-AUC | Brier Score | Recall@0.50 | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Champion)** | **0.7087 ± 0.0107** | **0.2921 ± 0.0140** | **0.2040** | **61.8%** | **0.7151** | **0.3023** |
| **Logistic Regression** | 0.7100 ± 0.0080 | 0.2935 ± 0.0108 | 0.2152 | 63.4% | 0.7124 | 0.2985 |
| **XGBoost** | 0.7030 ± 0.0110 | 0.2857 ± 0.0129 | 0.2013 | 58.3% | 0.7072 | 0.2918 |
| **Random Forest** | 0.6966 ± 0.0109 | 0.2770 ± 0.0078 | 0.1898 | 51.5% | 0.7001 | 0.2842 |

### 💰 Cost-Sensitive Underwriting Loss Optimization

Underwriting carries an asymmetric financial penalty where approving a bad loan is **10x more costly** than rejecting a solvent applicant:
- **Cost of False Negative ($C_{FN}$)**: $\approx \$5,000$ (unrecovered principal upon charge-off)
- **Cost of False Positive ($C_{FP}$)**: $\approx \$500$ (lost net interest margin and customer acquisition cost)

```text
Decision Strategy Comparison on Held-Out Test Set (7,716 loans):
┌──────────────────────────────┬───────────┬────────────────┬────────────────────────┐
│ Underwriting Strategy        │ Threshold │ Aggregate Loss │ Dollar Savings         │
├──────────────────────────────┼───────────┼────────────────┼────────────────────────┤
│ Unhedged (Approve All)       │    1.00   │  $5,625,000.00 │ Baseline               │
│ Standard Cutoff (Naive 0.50) │    0.50   │  $3,181,000.00 │ $2,444,000.00 (43.4%)  │
│ Cost-Optimal Policy (Tau*)   │    0.37   │  $2,770,000.00 │ $2,855,000.00 (50.8%)  │
└──────────────────────────────┴───────────┴────────────────┴────────────────────────┘
```
> By shifting from the naive 0.50 threshold to the cost-optimal **0.370** threshold, the institution captures **86.22% of all defaults**, eliminating **$2.85M** in portfolio write-offs.

## 🖥️ Running the Interactive Underwriting Dashboard

Launch the interactive Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

### Dashboard Features:
1. **Single Applicant Underwriting**: Real-time scoring, decision banners (`ACCEPT` / `REJECT`), risk tier categorization, and expected dollar loss.
2. **Regulatory Adverse Action Notices (FCRA Compliant)**: Automatically translates top positive SHAP attributions into legally compliant denial reason statements.
3. **Local SHAP Feature Attributions**: Interactive horizontal bar chart visualizing exact factors increasing and mitigating default hazard.
4. **Batch Portfolio Evaluation**: Upload or test loan books with summary acceptance rates, risk distributions, and CSV export.
5. **Model Economics Explorer**: Interactive cost curve and 5-fold CV leaderboard.

## 🏋️ Model Training & CLI Optimization

Run training and Bayesian hyperparameter tuning directly from the command line:

```bash
# Standard training and serialization
python src/train.py

# With Optuna hyperparameter optimization (25 trials)
python src/train.py --tune --trials 25
```

## 📓 Guided Step-by-Step Notebooks

1. [`01_data_cleaning.ipynb`](file:///notebooks/01_data_cleaning.ipynb): Data ingestion, right-censoring filtering, and target leakage elimination.
2. [`02_eda.ipynb`](file:///notebooks/02_eda.ipynb): Credit risk diagnostics, grade monotonicity, and leverage distributions.
3. [`03_feature_engineering.ipynb`](file:///notebooks/03_feature_engineering.ipynb): Domain ratios (`loan_to_income`, `installment_to_income`, `credit_history_years`) and scikit-learn `ColumnTransformer`.
4. [`04_modeling.ipynb`](file:///notebooks/04_modeling.ipynb): 5-fold CV benchmarking, cost-sensitive threshold curves, and LightGBM tuning.
5. [`05_explainability.ipynb`](file:///notebooks/05_explainability.ipynb): Global SHAP beeswarm plots, local waterfall attributions, and adverse action disclosures.

