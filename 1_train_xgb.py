#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:41:47 2024

@author: shikhar
"""
!pip install xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

# -----------------------
# Step 1: Load the Data
# -----------------------
data = pd.read_csv("/Users/shikhar/Documents/Ship2MyID/synthetic_data.csv")

# -----------------------
# Step 2: Inspect & Preprocess
# -----------------------
# Print the first few rows to verify data loading
print("Data Head:")
print(data.head())

# Identify the target variable
target_column = "Purchased"

# Features are all columns except the target
# Some columns are purely identifiers (Deal_ID, User_ID), we can exclude them as well.
exclude_cols = ["Deal_ID", "User_ID", target_column]

feature_cols = [c for c in data.columns if c not in exclude_cols]

# Separate features and target
X = data[feature_cols]
y = data[target_column]

# -----------------------
# Step 3: Encode Categorical Variables
# -----------------------
# Identify categorical features
# For simplicity, treat any non-numeric columns as categorical
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Convert categorical features using get_dummies (one-hot encoding)
X_encoded = pd.get_dummies(X, columns=cat_cols)

# -----------------------
# Step 4: Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# -----------------------
# Step 5: Train the XGBoost Model
# -----------------------
# Create the classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'  # For newer XGBoost versions, specify this explicitly
)

# Fit the model
model.fit(X_train, y_train)

# -----------------------
# Step 6: Evaluation
# -----------------------
# Predict on test set
y_pred = model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))

# Print confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------
# Step 7: Feature Importance (Optional)
# -----------------------
# Display feature importances
feature_importances = model.feature_importances_
feature_names = X_encoded.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df.head(10))

