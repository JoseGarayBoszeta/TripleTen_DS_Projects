# Project: Car Price Prediction for 'Rusty Bargain'

## 1. Project Goal

The goal of this project is to develop a machine learning model for the "Rusty Bargain" service, which is buying and selling used cars. The model must accurately predict the market value of a car based on its features.

The company has three key requirements for the final model:
1.  **Accuracy:** The model must achieve an **RMSE (Root Mean Squared Error) below 2500**.
2.  **Training Speed:** The model must train quickly to allow for frequent re-training on new data.
3.  **Prediction Speed:** The model must be able to make predictions very quickly to support the business application.

## 2. Data

The project uses a dataset of historical car sales, which includes the following key features:
* **VehicleType**, **Gearbox**, **FuelType**, **Brand**, **Model** (Categorical features)
* **RegistrationYear**, **Power**, **Kilometer** (Numerical features)
* **NotRepaired** (Categorical, indicating if the car has un-repaired damage)
* **Price** (Numerical, the target variable)

## 3. Methodology

### A. Data Preparation
1.  **Data Cleaning:**
    * Irrelevant columns (`DateCrawled`, `DateCreated`, `LastSeen`, `PostalCode`) were dropped.
    * Anomalous data was cleaned (e.g., `RegistrationYear` outside a realistic range, `Power` values that were too low or too high).
    * Duplicates were removed.
2.  **Handling Missing Values:** Missing values in key categorical features (`VehicleType`, `Gearbox`, `Model`, `FuelType`, `NotRepaired`) were filled with an 'unknown' placeholder or the mode.
3.  **Feature Encoding:**
    * For the tree-based boosting models (LightGBM, CatBoost), **Ordinal Encoding** was used. This method is fast and effective for these types of models.
    * For the baseline Linear Regression model, **One-Hot Encoding (OHE)** was used to convert categorical features into a format it can understand.
4.  **Data Splitting:** The data was split into training and test sets.



### B. Model Training & Evaluation
Several different regression models were trained and compared based on the project's three key criteria (RMSE, training time, and prediction time):

1.  **Linear Regression:** Trained as a fast, simple baseline to compare against.
2.  **LightGBM (LGBM) Regressor:** A powerful and extremely fast gradient boosting model.
3.  **CatBoost Regressor:** Another advanced gradient boosting model, known for its ability to natively handle categorical features.

All models were first trained with their default parameters to get an initial performance baseline.

### C. Hyperparameter Tuning
The most promising model, **LightGBM**, was selected for hyperparameter tuning to maximize its accuracy. `GridSearchCV` was used to systematically search for the best combination of parameters (like `n_estimators`, `learning_rate`, and `num_leaves`) to achieve the lowest possible RMSE.

## 4. Conclusion

The models were compared, and the final results were:
* **Linear Regression:** Had a high RMSE, failing to meet the accuracy requirement.
* **CatBoost:** Provided good accuracy (meeting the RMSE target) but was significantly slower to train.
* **LightGBM (Tuned):** This model was the clear winner. It achieved the **best RMSE score (well below the 2500 target)** and was by far the **fastest** in both training and prediction.

**Recommendation:**
The final **tuned LightGBM model** is recommended for the "Rusty Bargain" service. It successfully meets all project requirements: it is highly accurate, trains extremely quickly, and delivers near-instant predictions.

## 5. Key Libraries and Tools
* **Pandas & NumPy:** For data loading, cleaning, and manipulation.
* **Matplotlib & Seaborn:** For data visualization.
* **Scikit-learn:** For data preprocessing (`train_test_split`, `OrdinalEncoder`, `OneHotEncoder`), model training (`LinearRegression`), and hyperparameter tuning (`GridSearchCV`).
* **LightGBM:** For the `LGBMRegressor` model.
* **CatBoost:** For the `CatBoostRegressor` model.
