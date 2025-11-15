# Project: Gold Recovery Prediction for 'Zyfra'

## 1. Project Goal

The objective of this project is to develop a machine learning model for the 'Zyfra' mining company. The model must predict two key values:
1.  **Rougher output recovery:** The efficiency of the initial flotation stage.
2.  **Final output recovery:** The total, final efficiency of the gold recovery.

An accurate model will help 'Zyfra' optimize its production process, reduce costs, and operate more efficiently.

## 2. The Success Metric: sMAPE

This project uses a specific, custom-defined metric called **sMAPE** (symmetric Mean Absolute Percentage Error).

The sMAPE value for the rougher and final stages is calculated, and then combined into a single, final weighted score:

**$$Final\_sMAPE = 0.25 \times sMAPE(rougher) + 0.75 \times sMAPE(final)$$**

The sMAPE formula for a single set of predictions is:
**$$sMAPE = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2}$$**
where:
* $y_i$ is the true value.
* $\hat{y}_i$ is the predicted value.
* $N$ is the number of samples.

A custom function for this metric was built to be used with Scikit-learn.

## 3. Data

The project uses three datasets, all of which contain time-series data:
* `gold_recovery_train.csv`: The training set, which includes all features and the two target variables (`rougher.output.recovery` and `final.output.recovery`).
* `gold_recovery_test.csv`: The test set. It contains only the features that would be available at the time of prediction (i.e., it is missing intermediate and final outputs).
* `gold_recovery_full.csv`: The complete source dataset with all features and targets.



## 4. Methodology

### A. Data Exploration & Preprocessing
1.  **Verify Recovery Calculation:** The `rougher.output.recovery` calculation in the training set was checked against the provided formula and confirmed to be correct.
2.  **Align Features:** The training set was modified to *only* include features that were also present in the test set. This prevents data leakage and simulates a real-world prediction scenario.
3.  **Handle Missing Values (NaNs):** A significant number of missing values were present. Because this is time-series data (where a sensor reading is valid until the next one), the **forward fill (`ffill`)** method was used to propagate the last known valid observation forward.

### B. Model Development & Evaluation
Two separate models were trained: one to predict `rougher.output.recovery` and one for `final.output.recovery`.

1.  **Custom Scorer:** A custom `sMAPE` scorer was created using Scikit-learn's `make_scorer` function to be used in cross-validation.
2.  **Model Comparison:** Three different regression models were evaluated using **k-fold cross-validation** to find the one with the lowest sMAPE:
    * **Linear Regression**
    * **Decision Tree Regressor**
    * **Random Forest Regressor**
3.  **Baseline Model:** A `DummyRegressor` (predicting the mean) was used as a sanity check. All three models performed significantly better than this baseline.

## 5. Conclusion

The **Random Forest Regressor** consistently produced the **lowest (best) sMAPE scores** for both the rougher and final recovery targets during cross-validation.

* This model was selected as the final, most promising model.
* It was then trained on the full training dataset and evaluated on the hold-out test set to produce the final project sMAPE score, demonstrating its effectiveness in predicting gold recovery.

## 6. Key Libraries and Tools
* **Pandas & NumPy:** For data loading, cleaning, and manipulation.
* **Matplotlib & Seaborn:** For data visualization.
* **Scikit-learn:** For data preprocessing, `make_scorer`, `cross_val_score`, and for all models (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `DummyRegressor`).
