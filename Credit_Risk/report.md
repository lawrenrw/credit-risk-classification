Loan Risk Prediction Model Analysis
Overview of the Analysis

This analysis aimed to build a machine learning model to predict the risk level of loans based on financial data. The primary goal was to identify high-risk loans (1) versus healthy loans (0) using various borrower-related financial features. Predicting high-risk loans can help financial institutions mitigate default risks and make informed lending decisions.

The dataset included the following features:

    loan_size

    interest_rate

    borrower_income

    debt_to_income

    num_of_accounts

    derogatory_marks

    total_debt

The target variable was:

    loan_status: where 0 indicates a healthy loan and 1 indicates a high-risk loan.

A quick analysis of the target values showed a significant class imbalance:

y.value_counts()

This revealed that the majority of loans were labeled as healthy (0), with only a small portion flagged as high-risk (1).
Machine Learning Process

The analysis followed these steps:

    Loaded and explored the dataset.

    Separated the data into features (X) and labels (y).

    Split the data into training and testing sets using train_test_split.

    Trained a logistic regression model using LogisticRegression from scikit-learn.

    Evaluated the model using accuracy, precision, recall, and F1-score, particularly focusing on performance for high-risk loan predictions.

Results
Machine Learning Model 1: Logistic Regression

    Accuracy: 0.99

    Precision:

        Healthy Loan (0): 1.00

        High-Risk Loan (1): 0.84

    Recall:

        Healthy Loan (0): 0.99

        High-Risk Loan (1): 0.94

    F1-Score:

        Healthy Loan (0): 1.00

        High-Risk Loan (1): 0.89

Summary

The logistic regression model demonstrated outstanding performance overall, achieving 99% accuracy. It performed exceptionally well for predicting healthy loans (0) with perfect precision and near-perfect recall. Importantly, it also performed strongly on high-risk loans (1), with a recall of 94% and a respectable precision of 84%.
Recommendation:

    This model is a good candidate for identifying high-risk loans, especially in use cases where recall is more important than precision (e.g., better to flag too many risky loans than to miss any).

    However, if precision for high-risk loans is more important (e.g., avoiding false positives that might unfairly deny a borrower), further tuning or a different algorithm (like Random Forest with class weighting) may improve results.

At this stage, the logistic regression model is recommended as a strong baseline model. Further work could include exploring more complex models or addressing class imbalance with techniques like SMOTE or class weighting.