# README

This repository contains code for a machine learning project focused on loan prediction. The goal of the project is to build a model that can predict whether a loan application will be approved or not.

## Prerequisites

Make sure you have the following libraries installed:

- NumPy (imported as np)
- Pandas (imported as pd)
- Scikit-learn (imported modules: train_test_split, RandomizedSearchCV, GridSearchCV, StandardScaler, SVC, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, accuracy_score, roc_auc_score, LogisticRegression)
- Matplotlib (imported as plt)
- XGBoost (imported as xgb)

```

## Data

The training data for this project is stored in a CSV file named 'train_csv.csv'. You should place the file in the same directory as the code files.

## Data Preprocessing

The code begins with data preprocessing steps to handle missing values and convert categorical variables into numerical form. The missing values are filled using appropriate strategies:

- 'Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History' columns are filled with the mode (most frequent value) of the respective columns.
- 'LoanAmount' is filled with the mean of the non-null values.
- 'Loan_Amount_Term' is filled with the mode of the column.

The 'Loan_Status' column is mapped to binary values: 'N' is mapped to 0 and 'Y' is mapped to 1.

Categorical variables are one-hot encoded using the pd.get_dummies() function.

## Model Building

The dataset is split into training and testing sets using train_test_split() from scikit-learn. The training set is further used for model training and evaluation.

### XGBoost

An XGBoost classifier is trained using RandomizedSearchCV to find the best hyperparameters. The parameter grid consists of 'n_estimators', 'max_depth', 'learning_rate', and 'colsample_bytree' parameters.

The best parameters found by RandomizedSearchCV are printed, and the model is evaluated on the testing set by calculating the accuracy.

### Random Forest

A Random Forest classifier is trained using RandomizedSearchCV to find the best number of estimators ('n_estimators').

The best parameters found by RandomizedSearchCV are printed, and the model is evaluated on the testing set by calculating the accuracy.

### Decision Tree

A Decision Tree classifier is trained using GridSearchCV to find the best hyperparameters. The parameter grid consists of 'max_depth', 'min_samples_leaf', 'min_samples_split', and 'criterion' parameters.

The best parameters found by GridSearchCV are printed, and the model is evaluated on the testing set by calculating the accuracy.

### Support Vector Machine (SVM)

An SVM classifier is trained using RandomizedSearchCV to find the best hyperparameters. The parameter grid consists of 'kernel' and 'C' parameters.

The best parameters found by RandomizedSearchCV are printed, and the model is evaluated on the testing set by calculating the accuracy.

## Feature Importance

The code includes a function named 'feature_imp()' that calculates and plots the feature importances for the Decision Tree, Random Forest, and XGBoost models. The feature importances are displayed in descending order of importance.

## Note

Please ensure that you have the 'train_csv.csv' file in the same directory as the code files before running the code.

Feel free to modify the code and experiment with different models and hyperparameters to improve the performance of the loan prediction model.
