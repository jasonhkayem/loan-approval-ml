"""
train.py — end-to-end loan approval prediction pipeline.

Run with:
    python train.py

Pipeline stages (in order):
    1.  Load & inspect
    2.  Stratified train/test split
    3.  Imputation   (fit on train, apply to test — no leakage)
    4.  Encoding
    5.  Feature engineering
    6.  Scaling      (fit on train, apply to test — no leakage)
    7.  Anomaly detection  (training set only)
    8.  Baseline model
    9.  Model comparison with cross-validation
    10. Hyperparameter tuning
    11. Final model selection (data-driven: best CV AUC wins)
    12. Test-set evaluation
    13. SHAP interpretability
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import shap
from scipy.stats import randint
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from pyod.models.knn import KNN

from src.evaluate import (
    missing_values_table,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    print_cv_results,
    visualize_outliers,
)

DATA_FILE = Path(__file__).resolve().parent / "data" / "Loan_Prediction_Dataset.csv"


# Pipeline steps

def _impute(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fill missing values. All fill statistics are derived from X_train.

    Strategy per column:
    - Categorical (Credit_History, Self_Employed, Gender, Dependents, Married): mode
    - LoanAmount: mean — low missing rate (3.3%), mean is robust enough
    - Credit_History: mode is always 1.0 (84% of rows), which biases toward
      approval. Known limitation: 8.1% missing on the most predictive feature.
    - Loan_Amount_Term: KNN (k=5) — discrete valid terms (12–480 months);
      KNN picks a plausible neighbour value; mean could produce an impossible term.
    """
    X_train, X_test = X_train.copy(), X_test.copy()

    for col in ["Credit_History", "Self_Employed", "Gender", "Dependents", "Married"]:
        fill_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(fill_val)
        X_test[col]  = X_test[col].fillna(fill_val)

    loan_mean = X_train["LoanAmount"].mean()
    X_train["LoanAmount"] = X_train["LoanAmount"].fillna(loan_mean)
    X_test["LoanAmount"]  = X_test["LoanAmount"].fillna(loan_mean)

    knn_imputer = KNNImputer(n_neighbors=5)
    X_train[["Loan_Amount_Term"]] = knn_imputer.fit_transform(X_train[["Loan_Amount_Term"]])
    X_test[["Loan_Amount_Term"]]  = knn_imputer.transform(X_test[["Loan_Amount_Term"]])

    return X_train, X_test


def _encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    - Dependents: ordinal mapping ('3+' → 3). Order 0 < 1 < 2 < 3 is meaningful.
    - All other categoricals: one-hot with drop_first=True to avoid the dummy
      variable trap (perfect multicollinearity with an intercept).
    - Test columns are aligned to training columns to handle unseen categories.
    """
    X_train, X_test = X_train.copy(), X_test.copy()

    X_train["Dependents"] = X_train["Dependents"].replace("3+", 3).astype(int)
    X_test["Dependents"]  = X_test["Dependents"].replace("3+", 3).astype(int)

    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        dummies_train = pd.get_dummies(X_train[col], prefix=col, drop_first=True).astype(int)
        dummies_test  = pd.get_dummies(X_test[col],  prefix=col, drop_first=True).astype(int)
        dummies_test  = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)
        X_train = pd.concat([X_train.drop(columns=[col]), dummies_train], axis=1)
        X_test  = pd.concat([X_test.drop(columns=[col]),  dummies_test],  axis=1)

    return X_train, X_test


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-informed features for lending risk. Call after imputation, before scaling.

    - Total_Income: true household repayment capacity
    - Log transforms: compress right-skewed income/loan distributions
    - EMI: monthly instalment proxy (loan amount / term in months)
    - BalanceIncome: disposable income after servicing the loan
    - Loan_Income_Ratio: debt-to-income ratio — classic lending risk signal
    """
    df = df.copy()
    df["Total_Income"]      = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_log"]    = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"]   = np.log1p(df["Total_Income"])
    df["EMI"]               = df["LoanAmount"] / df["Loan_Amount_Term"]
    df["BalanceIncome"]     = df["Total_Income"] - (df["EMI"] * 1000)
    df["Loan_Income_Ratio"] = df["LoanAmount"] / df["Total_Income"]
    return df


def _scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    StandardScaler fit on training data only.
    Binary-encoded columns (0/1) are excluded — scaling them adds no value.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    cols = [
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Total_Income", "Loan_Income_Ratio", "LoanAmount_log", "TotalIncome_log",
        "EMI", "BalanceIncome",
    ]
    cols = [c for c in cols if c in X_train.columns]
    scaler = StandardScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols]  = scaler.transform(X_test[cols])
    return X_train, X_test


# Main pipeline

def main() -> None:
    # 1. Load 
    df = pd.read_csv(DATA_FILE)
    df.drop(columns=["Loan_ID"], inplace=True)
    print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")
    print(df.head())

    # 2. Feature / target split
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Encode target to 0/1 so sklearn metrics (AUC-ROC, PR) work out of the box.
    # LabelEncoder sorts alphabetically: N→0, Y→1 (Denied=0, Approved=1).
    le = LabelEncoder()
    y  = pd.Series(le.fit_transform(y), index=y.index, name="Loan_Status")

    counts = y.value_counts()
    print(f"\nClass distribution (0=Denied, 1=Approved):\n{counts}")
    print(f"Approval rate: {counts[1] / len(y):.1%}")
    print(
        "\nNote: ~69% approval rate → mild class imbalance."
        "\nAll models will use class_weight='balanced' and AUC-ROC as primary metric."
    )

    # 3. Stratified train/test split
    # stratify=y preserves the 69/31 class ratio in both sets.
    # Critical with only 614 rows — an unstratified split can skew the ratio.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

    # 4. Imputation
    print("\n── Missing values (training set) ──")
    print(missing_values_table(X_train))
    X_train, X_test = _impute(X_train, X_test)

    # 5. Encoding
    X_train, X_test = _encode(X_train, X_test)

    # 6. Feature engineering
    X_train = _engineer(X_train)
    X_test  = _engineer(X_test)
    print(f"\nFeature count after engineering: {X_train.shape[1]}")

    # 7. Scaling
    # Scaling happens BEFORE anomaly detection so that high-magnitude income
    # values (range 150–81 000) don't dominate distance calculations in KNN
    # and IsolationForest, producing biased outlier scores.
    X_train, X_test = _scale(X_train, X_test)

    # 8. Anomaly detection (training set only)
    # Three detectors vote independently. A row is removed only if ≥2 agree
    # it is an outlier — ensemble consensus reduces false positives.
    # We never filter the test set: in production you can't refuse to score
    # a loan application, and doing so here would be data leakage.
    print("\n── Anomaly Detection ──")
    pca      = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(X_train)

    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso.fit(X_train)
    iso_preds = (iso.predict(X_train) == -1).astype(int)  # IsolationForest uses -1 for outliers
    visualize_outliers(pca_data, iso_preds, "Isolation Forest")

    knn_dist = KNN(n_neighbors=10, method="largest", contamination=0.1)
    knn_dist.fit(X_train)
    visualize_outliers(pca_data, knn_dist.labels_, "KNN — Distance-Based")

    knn_dens = KNN(n_neighbors=10, method="mean", contamination=0.1)
    knn_dens.fit(X_train)
    visualize_outliers(pca_data, knn_dens.labels_, "KNN — Density-Based")

    votes = pd.DataFrame(
        {"iso": iso_preds, "knn_dist": knn_dist.labels_, "knn_dens": knn_dens.labels_},
        index=X_train.index,
    )
    rows_to_drop  = votes[votes.sum(axis=1) >= 2].index
    X_train_clean = X_train.drop(index=rows_to_drop)
    y_train_clean = y_train.drop(index=rows_to_drop)
    print(
        f"Removed {len(rows_to_drop)}/{len(X_train)} training rows "
        f"({len(rows_to_drop)/len(X_train):.1%}) flagged by ≥2 detectors."
    )

    # 9. Baseline
    # Always predicts the majority class (Approved). Accuracy ≈ 69%.
    # Any real model must beat this to be considered useful — this is our floor.
    print("\n── Baseline: Majority-Class Predictor ──")
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train_clean, y_train_clean)
    print(classification_report(y_test, dummy.predict(X_test), target_names=["Denied", "Approved"], zero_division=0))

    # 10. Model comparison with cross-validation
    # class_weight='balanced' compensates for the 69/31 class imbalance by
    # up-weighting the minority class (Denied) during training.
    # CV scoring = roc_auc: threshold-independent and robust to class imbalance.
    # (Accuracy would be misleading — a model that always says "Approved"
    #  gets 69% accuracy but is useless for predicting denials.)
    print("\n── 5-Fold Cross-Validation (ROC-AUC) ──")
    candidate_models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42),
    }
    print_cv_results(candidate_models, X_train_clean, y_train_clean, cv=5, scoring="roc_auc")

    # 11. Hyperparameter tuning
    print("\n── Hyperparameter Tuning ──")

    # RF: large, non-convex space → RandomizedSearchCV (50 draws; grid would be too slow)
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_distributions={
            "n_estimators":      randint(200, 800),
            "max_depth":         randint(4, 50),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf":  randint(1, 10),
            "max_features":      ["sqrt", "log2", None],
            "bootstrap":         [True, False],
        },
        n_iter=50, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1, verbose=0,
    )
    rf_search.fit(X_train_clean, y_train_clean)
    print(f"Best RF params : {rf_search.best_params_}")
    print(f"Best RF CV AUC : {rf_search.best_score_:.4f}")

    # LR: small, convex space → exhaustive GridSearchCV
    lr_search = GridSearchCV(
        LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42),
        param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100]},
        cv=5, scoring="roc_auc",
    )
    lr_search.fit(X_train_clean, y_train_clean)
    print(f"Best LR params : {lr_search.best_params_}")
    print(f"Best LR CV AUC : {lr_search.best_score_:.4f}")

    # 12. Final model — let the data decide
    # Pick the tuned model with the higher CV AUC.
    # Using CV score (not test score) for selection avoids optimising for the
    # specific test split, which would be a form of leakage.
    if rf_search.best_score_ >= lr_search.best_score_:
        final_model  = rf_search.best_estimator_
        winner_label = f"Tuned Random Forest (CV AUC {rf_search.best_score_:.4f})"
    else:
        final_model  = lr_search.best_estimator_
        winner_label = f"Tuned Logistic Regression (CV AUC {lr_search.best_score_:.4f})"
    print(f"\nFinal model: {winner_label}")

    # 13. Test-set evaluation
    # Evaluated on the held-out test set exactly once — after all modelling
    # decisions are locked in. Peeking at the test set earlier would inflate
    # reported performance.
    print("\n── Test-Set Results ──")
    y_pred = final_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Denied", "Approved"]))

    plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix — {winner_label}")
    roc_auc = plot_roc_curve(final_model, X_test, y_test, label=winner_label)
    ap      = plot_precision_recall_curve(final_model, X_test, y_test, label=winner_label)
    print(f"Test AUC-ROC: {roc_auc:.4f} | Average Precision: {ap:.4f}")

    # 14. SHAP interpretability
    # TreeExplainer for tree-based models: computes exact Shapley values using
    # the tree structure. LinearExplainer would be correct for Logistic Regression.
    print("\n── SHAP Feature Importance ──")
    if hasattr(final_model, "estimators_"):
        explainer = shap.TreeExplainer(final_model)
        shap_vals = explainer.shap_values(X_test)
        # Binary RF returns [class0_vals, class1_vals]; index 1 = Approved
        plot_shap = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    else:
        explainer = shap.LinearExplainer(final_model, X_train_clean, feature_perturbation="interventional")
        plot_shap = explainer(X_test).values

    shap.summary_plot(plot_shap, X_test, plot_type="violin", show=True)
    shap.summary_plot(plot_shap, X_test, plot_type="bar",    show=True)


if __name__ == "__main__":
    main()
