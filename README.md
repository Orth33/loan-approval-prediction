# Loan Approval Prediction System 🏦

This repository contains a comprehensive Machine Learning pipeline designed to predict whether a loan application will be **Approved** or **Rejected** based on financial and demographic features. This project was developed as part of a Data Science internship.


## 📌 Project Objectives
The task followed a structured data science workflow:
1. **Model Building:** Develop a classification system for loan risk assessment.
2. **Data Cleaning:** Handle missing values and perform categorical encoding.
3. **Imbalance Management:** Address class imbalance to ensure fair prediction for minority classes (Rejected loans).
4. **Performance Evaluation:** Optimize the model using **Precision, Recall, and F1-Score** rather than just Accuracy.


## 📊 Dataset Overview
The project utilizes the [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) from Kaggle.

- **Total Records:** 4,269
- **Target Variable:** `loan_status` (Approved/Rejected)
- **Key Features:** - `cibil_score`: Credit score of the applicant.
  - `income_annum`: Annual income.
  - `loan_amount`: Total amount requested.
  - `residential_assets_value`, `commercial_assets_value`, etc.


## 🛠️ Implementation Details

### 1. Data Preprocessing
- **Missing Values:** Imputed using median for numerical data and mode for categorical data.
- **Categorical Encoding:** Applied **Label Encoding** for the target variable and **One-Hot Encoding** for features like Education and Employment status.
- **Scaling:** Used `StandardScaler` to normalize feature ranges for better model convergence.

### 2. Handling Class Imbalance
Because the dataset is naturally imbalanced, I implemented **SMOTE (Synthetic Minority Over-sampling Technique)**. This technique generates synthetic examples for the minority class, preventing the model from being biased toward the majority class.

### 3. Model Comparison
I compared two distinct algorithms to find the best fit for this financial use case:
- **Logistic Regression:** A linear approach used as a baseline.
- **Decision Tree Classifier:** A non-linear, rule-based approach.


## 📈 Model Performance
Using the **Decision Tree Classifier** on the test set, the following results were achieved:

| Metric    | Score |
| :-------- | :---- |
| **Precision** | 0.94  |
| **Recall** | 0.97  |
| **F1-Score** | 0.96  |
| **Accuracy** | 0.97  |

*Note: High Recall indicates the model is excellent at identifying nearly all potential loan approvals/rejections correctly.*


## 🚀 How to Use
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/Orth33/loan-approval-prediction.git
