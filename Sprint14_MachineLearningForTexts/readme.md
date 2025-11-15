# Project: Sentiment Analysis for 'Film Junky Union'

## 1. Project Goal

The "Film Junky Union," a community for movie enthusiasts, required a system to automatically filter and categorize movie reviews. The goal of this project was to build a machine learning model to classify reviews as either **positive (1)** or **negative (0)**.

The primary success metric was to achieve an **F1 score of at least 0.85** on the test set.

## 2. Data

The project used the **IMBD movie reviews dataset** (`imdb_reviews.tsv`).
* **`review`**: The full text of the movie review.
* **`pos`**: The target variable (0 for negative, 1 for positive).
* **`ds_part`**: A column indicating whether the review belongs to the 'train' or 'test' set.

## 3. Methodology

This project compared two distinct Natural Language Processing (NLP) approaches to find the best-performing model.

### A. Exploratory Data Analysis (EDA)
An initial analysis showed the dataset is **well-balanced** (almost 50/50 positive vs. negative), so no special class imbalance techniques were needed.

---

### B. Approach 1: Traditional ML with TF-IDF

1.  **Text Preprocessing:** The text was cleaned using NLTK:
    * Tokenized into individual words.
    * Lemmatized to its root form (e.g., "running" -> "run").
    * Cleaned of common stop words (e.g., 'the', 'is', 'a').
2.  **Vectorization:** The cleaned text was converted into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
3.  **Model Training:** Several models were trained on these TF-IDF vectors:
    * Logistic Regression
    * LightGBM (LGBM) Classifier

### C. Approach 2: Advanced ML with BERT

1.  **Embeddings:** Instead of TF-IDF, pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** embeddings were used. A utility function (`BERT_text_to_embeddings()`) was used to convert the raw review text directly into sophisticated contextual numerical vectors.

2.  **Model Training:** A **Logistic Regression** model was trained on the output BERT embeddings. Due to the computational cost of BERT, this process was noted as being significantly more resource-intensive than the TF-IDF approach.

## 4. Conclusion & Results

The performance of all models was evaluated on the test set.

* **TF-IDF Models:**
    * **Logistic Regression:** F1 Score ≈ 0.88
    * **LightGBM:** F1 Score ≈ 0.87

* **BERT Model:**
    * **Logistic Regression (on BERT embeddings):** F1 Score ≈ 0.93

**Winner:** The model using **BERT embeddings** was the clear winner, achieving the highest F1 score (0.93) and easily surpassing the project goal of 0.85.

This demonstrates that while TF-IDF is a strong baseline, the deep contextual understanding from BERT provides a significant boost in performance for this sentiment analysis task.

## 5. Key Libraries and Tools
* **Pandas:** For data loading and manipulation.
* **NLTK (Natural Language Toolkit):** For text preprocessing (lemmatization, stop words).
* **Scikit-learn (sklearn):** For `TfidfVectorizer`, `LogisticRegression`, and evaluation metrics.
* **LightGBM (LGBM):** For the `LGBMClassifier` model.
* **Transformers / Pytorch:** Libraries used to load the pre-trained BERT model and generate embeddings.
