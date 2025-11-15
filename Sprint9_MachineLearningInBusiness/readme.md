# Project: Profit and Risk Analysis for 'OilyGiant' Oil Wells

## 1. Project Goal

The goal of this project is to analyze geological data from three different regions to find the most profitable location for a new oil well. The mining company, 'OilyGiant', needs to select the region with the highest potential profit while minimizing the risk of financial loss.

This project involves:
1.  Training a **Linear Regression** model for each region to predict the volume of oil reserves in new wells.
2.  Calculating the potential profit for the top 200 wells in each region.
3.  Using the **Bootstrapping** technique to assess the distribution of profit and quantify the **risk of loss**.

The final recommendation must be for a region with a **risk of loss lower than 2.5%**.

## 2. Data

The data is provided in three separate files, one for each region: `geo_data_0.csv`, `geo_data_1.csv`, and `geo_data_2.csv`.

Each file contains 100,000 data points with the following features:
* `f0`, `f1`, `f2`: Geological features of the well.
* `product`: The actual volume of oil reserves in the well (in thousands of barrels). This is the **target variable**.

## 3. Methodology

### A. Model Training
1.  **Data Preparation:** The data for each of the three regions was loaded and prepared.
2.  **Train-Test Split:** Each region's dataset was split into a training set (75%) and a validation set (25%).
3.  **Model Training:** A separate **Linear Regression** model was trained for each region to predict the `product` (volume of reserves).
4.  **Evaluation:** The models were evaluated on their respective validation sets using the **Root Mean Squared Error (RMSE)** to measure prediction accuracy.

### B. Profit & Risk Calculation
This section simulates a real-world business scenario with the following constraints:
* **Budget:** 100 million for the development of 200 wells.
* **Revenue:** $4,500 per unit (1,000 barrels of oil).
* **Break-even:** The volume of reserves needed to break even was calculated to understand the baseline.

A function was created to calculate the total profit from the **top 200 predicted wells**. This function uses the *model's predictions* to select the wells but then calculates profit based on the *actual* reserve volumes of those selected wells.

### C. Risk Assessment with Bootstrapping
To understand the range of possible outcomes and the risk of loss, bootstrapping was performed 1,000 times for each region:
1.  From the validation set, 500 data points were randomly sampled (with replacement).
2.  The top 200 wells were selected based on the model's predictions for this sample.
3.  The profit for these 200 wells was calculated using the *actual* reserve values.
4.  This process was repeated 1,000 times to create a distribution of 1,000 potential profit outcomes.



## 4. Conclusion

The bootstrapping results provided a clear picture of the potential for each region:
* **Average Profit:** The mean profit from the 1,000 bootstrap samples was calculated for each region.
* **95% Confidence Interval:** A 95% confidence interval for profit was determined.
* **Risk of Loss:** The percentage of bootstrap samples that resulted in a negative profit (loss) was calculated.

**Recommendation:**
The analysis identified **Region 1 (`geo_data_1.csv`)** as the clear winner. This region not only had the **highest average profit** but was also the **only region to meet the business requirement of having a risk of loss under 2.5%**.

## 5. Key Libraries and Tools
* **Pandas & NumPy:** For data manipulation and calculations.
* **Matplotlib:** For visualization.
* **Scikit-learn:** For `LinearRegression`, `train_test_split`, and `mean_squared_error`.
* **SciPy:** For statistical functions (e.g., calculating confidence intervals).
