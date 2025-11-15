# Project: Predicting Customer Churn for 'Interconnect'

## 1. Project Goal

The primary goal of this project is to analyze customer data for the telecom provider 'Interconnect' and develop a machine learning model that can accurately predict customer churn. By identifying at-risk customers, the company can implement targeted retention strategies, which are more cost-effective than acquiring new customers.

The model's success will be evaluated based on its final **AUC-ROC score** on the test set.

## 2. Data

The dataset is comprised of four separate files, which were merged to create a complete customer profile:
* **`contract.csv`**: Information about the customer's contract type, payment method, monthly charges, and total charges. Also contains `BeginDate` and `EndDate`.
* **`personal.csv`**: Customer's demographic information (gender, age, partner, dependents).
* **`internet.csv`**: Details on the customer's internet service and add-ons (e.g., online security, tech support).
* **`phone.csv`**: Information on whether the customer has phone service.

## 3. Methodology

The project followed a structured machine learning workflow:

### A. Data Preprocessing & Merging
1.  **Data Loading:** All four data sources were loaded and inspected.
2.  **Merging:** The tables were merged into a single DataFrame using the unique `customerID`.
3.  **Data Cleaning:**
    * Irrelevant columns (`customerID`) were dropped.
    * Missing values (e.g., in `TotalCharges` for new customers) were identified and filled.
    * Data types were corrected (e.g., `TotalCharges` was converted to a numeric type).

### B. Feature Engineering
1.  **Target Variable (`is_churn`):** The `EndDate` column was used to create the binary target variable. Customers with a recent `EndDate` were marked as "churned" (1), and all others were marked as "retained" (0).
2.  **Tenure:** A key predictive feature, **`tenure_days`**, was engineered by calculating the time (in days) between the `BeginDate` and `EndDate` (or the present date for retained customers).
3.  **Encoding:** All categorical features (e.g., `PaymentMethod`, `InternetService`) were converted into a numerical format using One-Hot Encoding.
4.  **Scaling:** All numerical features (e.g., `MonthlyCharges`, `TotalCharges`, `tenure_days`) were scaled using `StandardScaler`.
5.  **Data Splitting:** The final dataset was split into training and test sets.

### C. Model Development & Evaluation
Several classification models were trained and evaluated to find the best performer:

1.  **Logistic Regression:** Trained as a fast, interpretable baseline model.
2.  **Random Forest Classifier:** An ensemble model trained to capture more complex, non-linear relationships.
3.  **LightGBM (LGBM) Classifier:** A powerful gradient-boosting model known for its high speed and accuracy.

The models were first compared, and then the most promising model(s) were selected for **hyperparameter tuning** (e.g., using `GridSearchCV`) to find the optimal combination of parameters and maximize the AUC-ROC score.

## 4. Feature Importance

After training the final model, a **feature importance analysis** was conducted. This step identifies which customer attributes are the strongest predictors of churn. Key findings from this analysis show that factors like **contract type, `tenure_days`, and total monthly charges** are the most significant drivers of whether a customer will stay or leave.

## 5. Conclusion

All models were evaluated on the held-out test set. The **LightGBM model**, after hyperparameter tuning, was identified as the top-performing model, achieving the highest AUC-ROC and accuracy.

This final model, combined with the insights from the feature importance analysis, provides 'Interconnect' with a powerful tool to identify customers at high risk of churn and understand the key reasons *why*.

## 5. Key Libraries and Tools
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Matplotlib & Seaborn:** For data visualization and feature analysis.
* **Scikit-learn (sklearn):** For data preprocessing, hyperparameter tuning (`GridSearchCV`), and building the Logistic Regression and Random Forest models.
* **LightGBM (LGBM):** For building the gradient-boosting classification model.
