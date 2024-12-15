import pandas as pd
import joblib

# -----------------------
# Step 1: Load the Saved Model
# -----------------------
# Load the serialized model
loaded_model = joblib.load('xgb_model.pkl')

# -----------------------
# Step 2: Load and Pre-process New Data
# -----------------------
# Load new data (ensure it has the same structure as the training data)
new_data = pd.read_csv("/Users/shikhar/Documents/Ship2MyID/synthetic_data_2.csv")

# Print the first few rows to verify
#print("New Data Head:")
#print(new_data.head())

# Match the pre-processing used during training
# Exclude identifier and target columns
exclude_cols = ["Deal_ID", "User_ID", "Purchased"]  # Replace 'Purchased' with the actual target column
feature_cols = [c for c in new_data.columns if c not in exclude_cols]

# Select the features
X_new = new_data[feature_cols]

# Identify categorical features
cat_cols = X_new.select_dtypes(include=['object']).columns.tolist()

# Perform one-hot encoding on categorical features
X_new_encoded = pd.get_dummies(X_new, columns=cat_cols)

# Align the columns with the training data
# This ensures the new data has the same feature structure as the training data
# Fill any missing columns with zeros (they weren't present in the new data)
trained_features = loaded_model.get_booster().feature_names
for col in trained_features:
    if col not in X_new_encoded:
        X_new_encoded[col] = 0

# Reorder columns to match the training data
X_new_encoded = X_new_encoded[trained_features]

# -----------------------
# Step 3: Make Predictions
# -----------------------
# Use the loaded model to predict on the new data
predictions = loaded_model.predict(X_new_encoded)

# Print predictions
# Create a DataFrame to display results in the desired format
output_df = new_data[['Deal_Category', 'Product_Category']].copy()  # Select relevant columns
output_df['Purchase Prediction'] = predictions  # Add predictions as a new column

# Print the results in tabular format
print("\nPurchase Prediction as a function of Deal Category & Product Category:")
print("\n")
print(output_df)