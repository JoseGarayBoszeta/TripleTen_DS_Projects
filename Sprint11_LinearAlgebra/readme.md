# Project: Protecting Client Data with Linear Algebra

## 1. Project Goal

The insurance company "Sure Tomorrow" needs a way to protect its clients' personal data. The goal of this project is to develop a data transformation method that "obfuscates" (hides) the original personal information (like age, gender, salary) without compromising the quality of a machine learning model.

The task is to prove that a specific linear algebra-based transformation can successfully hide the data while leaving the predictions of a **Linear Regression** model completely unchanged.

## 2. The Core Hypothesis & Proof

The method involves multiplying the original feature matrix ($X$) by a randomly generated, invertible matrix ($P$).

* **Original Features:** $X$
* **Transformed Features:** $Z = XP$

The hypothesis is that the model's predictions ($\hat{y}$) will be identical for both the original and transformed data.



### Mathematical Proof

The weights ($w$) for a linear regression model are found using the formula:
$$w = (X^T X)^{-1} X^T y$$
The predictions ($\hat{y}$) are:
$$\hat{y} = Xw = X(X^T X)^{-1} X^T y$$

---
For the transformed data $Z = XP$, the new weights ($w_P$) will be:
$$w_P = ((XP)^T (XP))^{-1} (XP)^T y$$

By applying matrix properties ($ (AB)^T = B^T A^T $ and $ (AB)^{-1} = B^{-1} A^{-1} $), we can simplify this equation:
$$w_P = (P^T (X^T X) P)^{-1} P^T X^T y$$
$$w_P = P^{-1} (X^T X)^{-1} (P^T)^{-1} P^T X^T y$$

Since $(P^T)^{-1} P^T$ cancels out to become the identity matrix ($I$), we are left with:
$$w_P = P^{-1} (X^T X)^{-1} X^T y$$

If we substitute the original weights $w = (X^T X)^{-1} X^T y$, we find:
**$$w_P = P^{-1} w$$**

---
Finally, we make predictions ($\hat{y}_P$) using the transformed features and weights:
$$\hat{y}_P = Z w_P = (XP) (P^{-1} w)$$

The matrices $P$ and $P^{-1}$ cancel each other out ($P P^{-1} = I$), leaving:
$$\hat{y}_P = X I w = Xw$$

This proves that:
**$$\hat{y}_P = \hat{y}$$**

The predictions from the transformed data are mathematically identical to the predictions from the original data.

## 3. Methodology

To test this proof in practice, the following steps were taken:

1.  **Load Data:** The insurance dataset (`insurance.csv`) was loaded, with features (`age`, `gender`, `salary`, `family_members`) and the target (`insurance_benefits`).
2.  **Baseline Model:** A Linear Regression model was trained on the **original, untransformed data**. Its $R^2$ (R-squared) score was calculated and saved.
3.  **Create Invertible Matrix:** A random square matrix ($P$) was created with dimensions matching the number of features (4x4). This matrix was checked to ensure it was invertible.
4.  **Transform Data:** The original feature matrix ($X$) was multiplied by the invertible matrix ($P$) to create the new, "obfuscated" feature matrix ($Z$).
5.  **Obfuscated Model:** A second Linear Regression model was trained on the **new, transformed data ($Z$)**.
6.  **Compare Results:** The $R^2$ score of the second model was compared to the $R^2$ score of the baseline model.

## 4. Conclusion

The $R^2$ scores for both the baseline model and the obfuscated model were **identical**.

This experiment successfully proved that multiplying a feature matrix by an invertible matrix **does not change the quality or predictions** of a Linear Regression model. This method is a valid and secure way for "Sure Tomorrow" to protect its clients' personal information while still using it for machine learning.

## 5. Key Libraries and Tools
* **Pandas:** For data loading and manipulation.
* **NumPy:** For linear algebra operations and creating the transformation matrix.
* **Scikit-learn:** For `LinearRegression` and calculating the `r2_score`.
