# Project: Predicting Customer Churn for Beta Bank

## 1. Project Goal

The primary goal of this project is to develop a machine learning model that can accurately predict whether a customer will leave Beta Bank. By identifying customers at high risk of churn, the bank can take proactive steps to retain them.

The key success metric is to achieve an **F1 score of at least 0.59** on the test set.

## 2. Data

The project uses a dataset (`Churn.csv`) containing historical data on Beta Bank's customers. The features include:
* Customer demographics (Geography, Gender, Age)
* Account information (CreditScore, Tenure, Balance, NumOfProducts)
* Activity metrics (HasCrCard, IsActiveMember, EstimatedSalary)
* **Target Variable:** `Exited` (1 if the customer churned, 0 if they remained)

## 3. Methodology

### A. The Challenge: Class Imbalance
A core challenge in this dataset is the **severe class imbalance**. Exploratory data analysis revealed that only ~20% of customers in the dataset had churned (class 1), while ~80% remained (class 0). A model trained on this imbalanced data will be biased towards predicting the majority class (retained) and will perform poorly on the F1 metric, which is crucial for this problem.



### B. Data Preparation
1.  **Data Cleaning:** Irrelevant columns (RowNumber, CustomerId, Surname) were dropped.
2.  **Feature Encoding:** Categorical features (`Geography`, `Gender`) were converted into numerical format using **One-Hot Encoding (OHE)**.
3.  **Data Splitting:** The data was split into three distinct sets: training (60%), validation (20%), and test (20%).
4.  **Scaling:** All numerical features were scaled using `StandardScaler` to ensure models were not biased by feature magnitudes.

### C. Model Training & Imbalance Handling
Three classification models were trained and evaluated:
* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**

Each model was tested using four different approaches to handle the class imbalance:
1.  **Baseline:** Trained on the original, imbalanced data.
2.  **Class Weighting:** Using the `class_weight='balanced'` parameter in the models to penalize mistakes on the minority class more heavily.
3.  **Upsampling:** Randomly duplicating samples from the minority class (churned) in the training set until it was balanced with the majority class.
4.  **Downsampling:** Randomly removing samples from the majority class (retained) in the training set until it was balanced with the minority class.

## 4. Conclusion

The models' F1 scores were compared on the validation set for each technique. The **Random Forest Classifier** trained with the **Upsampling** technique yielded the highest and most stable F1 score.

This final model was then evaluated on the hold-out test set.

**Final Model Performance (on Test Set):**
* **Model:** Random Forest Classifier
* **Technique:** Upsampling
* **F1 Score:** 0.60
* **AUC-ROC:** 0.85

This model **successfully surpassed the target F1 score of 0.59**, providing Beta Bank with a reliable tool to identify at-risk customers.

## 5. Key Libraries and Tools
* **Pandas & NumPy:** For data manipulation.
* **Scikit-learn (sklearn):** For data preprocessing (`StandardScaler`, `OHE`), model training, metrics (`f1_score`, `roc_auc_score`), and utilities (`resample`).
* **Matplotlib:** For data visualization.
