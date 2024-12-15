#Serialising the model as a joblib file
!pip install joblib

import joblib

# Save the trained model to a file
joblib.dump(model, 'xgb_model.pkl')