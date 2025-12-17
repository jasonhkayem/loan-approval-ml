# Loan Approval Prediction Using Machine Learning

## Project Overview
This project presents an end-to-end supervised machine learning workflow designed to predict loan approval decisions based on applicant demographic and financial attributes. The objective is to demonstrate applied machine learning practices, including data preprocessing, feature engineering, model comparison, and result interpretation, using a realistic credit risk dataset.

The project mirrors a real-world scenario commonly encountered in financial institutions when evaluating loan applications.

---

## Problem Statement
Financial institutions must evaluate loan applications efficiently while minimizing credit risk.

The goal of this project is to **predict whether a loan application will be approved or rejected** using historical applicant data and supervised machine learning models.

This task is formulated as a **binary classification problem**, where the target variable represents the loan approval status.

---

## Dataset Description
- **Domain:** Finance / Credit Risk  
- **Observations:** Approximately 600 loan applications  
- **Features:** Combination of numerical and categorical variables  
- **Target Variable:** `Loan_Status` (Approved / Not Approved)

### Feature Examples
- Applicant income  
- Co-applicant income  
- Loan amount and loan term  
- Credit history  
- Employment status  
- Education level  
- Property area  

The dataset contains missing values and skewed financial variables, reflecting realistic data quality challenges.

---

## Methodology

### Data Preprocessing
- Handling missing values for numerical and categorical features
- Encoding categorical variables
- Scaling numerical features for model compatibility
- Applying log transformations to skewed financial variables
- Splitting data into training and testing sets prior to modeling

---

### Feature Engineering
- Creation of derived income-related features
- Log transformation to stabilize variance in monetary variables
- Feature selection guided by domain relevance and model interpretability

---

### Models Implemented
The following supervised learning models were trained and compared:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  

This selection enables comparison between linear models, distance-based methods, and ensemble techniques.

---

## Model Evaluation
Models were evaluated using:
- Accuracy
- Confusion matrices
- Precision
- Recall
- F1-score

Ensemble and non-linear models (Random Forest and SVM) achieved the strongest overall performance, indicating their suitability for capturing complex relationships in financial data.

---

## Key Insights
- Credit history was consistently the most influential feature across models
- Income-related variables benefited significantly from log transformation
- Ensemble methods provided greater robustness compared to simpler classifiers
- Proper preprocessing had a substantial impact on predictive performance

---

## Limitations and Future Improvements
- The dataset size is relatively small and may limit generalization
- Class imbalance handling could be further explored
- Hyperparameter tuning could be expanded
- Model explainability techniques (e.g., SHAP) could improve interpretability

---

## Tech Stack
- **Programming Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Development Environment:** Jupyter Notebook  

---

## Project Structure
