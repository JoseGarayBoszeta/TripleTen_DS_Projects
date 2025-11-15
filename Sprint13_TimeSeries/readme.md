# Project: Taxi Order Prediction for 'Sweet Lift Taxi'

## 1. Project Goal

The "Sweet Lift Taxi" company needs to forecast the number of taxi orders for the next hour. The goal is to build a machine learning model that can make this prediction, which will help optimize driver dispatch and reduce wait times.

The success metric is to achieve a **Root Mean Squared Error (RMSE) of no more than 48** on the final test set.

## 2. Data

The project uses a historical dataset (`taxi.csv`) of taxi orders.
* **Timeframe:** Data is provided in 10-minute intervals.
* **Target:** `num_orders` (the number of orders).

## 3. Methodology

### A. Data Preparation
1.  **Loading:** The data was loaded, with the `datetime` column parsed and set as the index.
2.  **Resampling:** To match the project goal (predicting for the next hour), the 10-minute interval data was **resampled into 1-hour intervals**, summing the `num_orders` for each hour.

### B. Exploratory Data Analysis (EDA)
1.  **Stationarity:** The resampled time series was plotted.
2.  **Trends & Seasonality:** Rolling means and standard deviations were analyzed to identify patterns. The data clearly showed:
    * **Daily Seasonality:** Peaks and valleys corresponding to the time of day (e.g., morning/evening rush hours, lulls in the early morning).
    * **Weekly Seasonality:** Different patterns for weekdays versus weekends.
    * **Overall Trend:** A general upward trend in the number of orders over time.

### C. Feature Engineering
To allow the models to "see" the time-based patterns, the following features were engineered from the datetime index:
* Day of the Week
* Hour of the Day
* **Lag Features:** The number of orders from previous time steps (e.g., 1 hour ago, 2 hours ago, 24 hours ago). This helps the model understand the immediate past.
* **Rolling Mean (Sliding Window):** The average number of orders over a recent window (e.g., the last 6 hours) to capture the current trend.

### D. Model Training
The data was split **chronologically** into training, validation, and test sets to simulate a real-world forecasting scenario.

Several models were trained and evaluated on the validation set using RMSE:
1.  **Linear Regression:** As a fast baseline.
2.  **Random Forest Regressor:** To capture more complex, non-linear patterns.
3.  **LightGBM (LGBM) Regressor:** A gradient boosting model known for its high performance and speed.

## 4. Conclusion

The **LightGBM (LGBM) Regressor** was the top-performing model on the validation set.

This final model was then trained on the full training set and evaluated on the hold-out test set. It successfully achieved an **RMSE score below the 48 target**, meeting the project's success criteria. This model provides "Sweet Lift Taxi" with a valuable tool for forecasting demand.

## 5. Key Libraries and Tools
* **Pandas:** For data loading, resampling, and feature engineering.
* **NumPy:** For numerical operations.
* **Matplotlib:** For plotting the time series.
* **Statsmodels:** For time series decomposition and analysis.
* **Scikit-learn:** For data splitting, `LinearRegression`, and `RandomForestRegressor`.
* **LightGBM:** For the `LGBMRegressor` model.
