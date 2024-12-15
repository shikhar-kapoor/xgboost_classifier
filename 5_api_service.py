from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('xgb_model.pkl')

# Identify columns to exclude and known feature order from training
exclude_cols = ["Deal_ID", "User_ID", "Purchased"]  # adjust as necessary
trained_features = model.get_booster().feature_names

# Define input schema using Pydantic
class PredictionInput(BaseModel):
    Deal_ID: str
    User_ID: str
    Deal_Category: str
    Product_Category: str
    # Add all other columns present during training except 'Purchased'
    # For example:
    Day_Of_Week_Start: str
    Day_Of_Week_End: str
    Deal_Duration_Hours: float
    # ... add any additional columns from your original dataset

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the XGBoost Prediction API"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert input data into a DataFrame
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    
    # Separate features (excluding target and IDs)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_new = df[feature_cols]
    
    # Identify categorical features
    cat_cols = X_new.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical features
    X_new_encoded = pd.get_dummies(X_new, columns=cat_cols)
    
    # Ensure all trained features exist in new data
    for col in trained_features:
        if col not in X_new_encoded.columns:
            X_new_encoded[col] = 0
            
    # Reorder columns to match training data
    X_new_encoded = X_new_encoded[trained_features]

    # Make prediction
    prediction = model.predict(X_new_encoded)
    
    # Format the result
    # If prediction is a probability or a class, adjust accordingly.
    # Assuming binary classification here.
    result = {
        "Deal_Category": data_dict["Deal_Category"],
        "Product_Category": data_dict["Product_Category"],
        "Purchase Prediction": int(prediction[0])
    }
    
    return result
