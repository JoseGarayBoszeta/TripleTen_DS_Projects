# Project: Recommending Mobile Plans for 'Megaline'

## 1. Project Goal

The mobile carrier Megaline wants to develop a machine learning model that can analyze the behavior of its subscribers and recommend one of two newer plans: **Smart** or **Ultra**.

The goal is to build a classification model that achieves the **highest possible accuracy**, with a minimum threshold of **0.75 (75%)** on the test set.

## 2. Data

The project uses the `users_behavior.csv` dataset, which contains pre-processed monthly behavior data for subscribers who have already switched to the new plans.

* **Features:**
    * `calls`: number of calls
    * `minutes`: total call duration in minutes
    * `messages`: number of text messages
    * `mb_used`: Internet traffic used in MB
* **Target Variable:**
    * `is_ultra`: The plan for the current month (1 for Ultra, 0 for Smart)

## 3. Methodology

### A. Data Splitting
Since there was no pre-existing test set, the data was split into three distinct parts:
1.  **Training Set (60%):** Used to train the models.
2.  **Validation Set (20%):** Used to tune hyperparameters and select the best model.
3.  **Test Set (20%):** Held back until the very end to provide a final, unbiased evaluation of the chosen model.

### B. Model Training & Hyperparameter Tuning
Three different classification models were investigated:
1.  **Decision Tree Classifier:** A fast, simple model. Its `max_depth` hyperparameter was tuned to find the best balance between performance and overfitting.
2.  **Random Forest Classifier:** A more complex ensemble model. Its `n_estimators` (number of trees) and `max_depth` were explored to find the best combination.
3.  **Logistic Regression:** A linear model used as a strong baseline.

The models were trained on the training set and their accuracy was compared using the validation set.

### C. Sanity Check
A "sanity check" was performed to ensure the models were actually learning. The models' accuracy was compared to the accuracy of simply guessing the majority class, which confirmed that all trained models were performing significantly better than a random or naive guess.

## 4. Conclusion

The **Random Forest Classifier** was the top-performing model on the validation set, achieving the highest accuracy.

This final model was then evaluated on the hold-out **test set**. It achieved an accuracy score **above the 0.75 threshold**, successfully meeting the project's goal. This model is now ready to help Megaline recommend the most suitable plans to its customers.

## 5. Key Libraries and Tools
* **Pandas:** For data loading and manipulation.
* **Scikit-learn (sklearn):** For data splitting (`train_test_split`) and for all models (`DecisionTreeClassifier`, `RandomForestClassifier`, `LogisticRegression`).
