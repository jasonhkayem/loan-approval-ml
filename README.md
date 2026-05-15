# Loan Approval Prediction

Binary classification model that predicts whether a loan application will be approved, trained on 614 historical applications.

## Project Structure

```
├── data/
│   └── Loan_Prediction_Dataset.csv   # 614 rows, 12 features + target
├── notebooks/
│   └── eda.ipynb                     # Exploratory data analysis (not the source of truth)
├── src/
│   ├── __init__.py
│   └── evaluate.py                   # Metrics, plots, reporting helpers
├── train.py                          # End-to-end pipeline — run this
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python train.py
```

## Dataset

614 loan applications with features: applicant demographics, income, loan amount, credit history, and property area. Target: `Loan_Status` (Y = Approved, N = Denied).

**Class distribution:** ~69% Approved / 31% Denied.

## Pipeline

1. Stratified 80/20 train/test split
2. Missing value imputation (mode for categoricals, mean for LoanAmount, KNN for Loan_Amount_Term)
3. Categorical encoding (ordinal for Dependents, one-hot for others)
4. Feature engineering (6 derived features: Total_Income, log transforms, EMI, BalanceIncome, Loan_Income_Ratio)
5. StandardScaler on continuous features
6. Anomaly detection — ensemble of 3 detectors (IsolationForest + 2x KNN); consensus vote removes outliers from training only
7. Baseline: DummyClassifier (majority class)
8. Model comparison: Logistic Regression, Decision Tree, Random Forest — 5-fold CV scored by AUC-ROC
9. Hyperparameter tuning: RandomizedSearchCV (RF) + GridSearchCV (LR)
10. Final model: whichever tuned model achieves higher CV AUC
11. SHAP interpretability (TreeExplainer for RF, LinearExplainer for LR)

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `stratify=y` on split | Preserves 69/31 class ratio with only 614 rows |
| `class_weight='balanced'` | Prevents minority class (Denied) from being ignored |
| AUC-ROC as primary metric | Accuracy is misleading under class imbalance |
| Scale before anomaly detection | Prevents high-magnitude income values from biasing distance scores |
| Ensemble consensus for outlier removal | Reduces false positives; 3 detectors must agree (≥2 of 3) |
| CV score drives model selection | Test set is untouched until final evaluation — avoids leakage |
