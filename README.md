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
