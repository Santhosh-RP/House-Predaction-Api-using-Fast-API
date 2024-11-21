from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Create the FastAPI app
app = FastAPI()

# Load the model
try:
    # Load the model using joblib (adjust the path to your model file)
    model = joblib.load("Models/gradient_boosting_model.joblib")
except Exception as e:
    raise Exception(f"Model file not found or could not be loaded. Error: {str(e)}")

# Define the request model for input data
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float
    feature10: float
    feature11: float
    feature12: float
    feature13: float

# Root endpoint to check if the app is running
@app.get("/")
async def read_root():
    return {"message": "FastAPI is running!"}

# Prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Extract features from request
        features = [
            request.feature1, request.feature2, request.feature3, 
            request.feature4, request.feature5, request.feature6, 
            request.feature7, request.feature8, request.feature9, 
            request.feature10, request.feature11, request.feature12, 
            request.feature13
        ]
        
        # Reshape for a single prediction
        input_data = np.array(features).reshape(1, -1)
        
        # Perform prediction
        prediction = model.predict(input_data)
        
        # Return prediction result
        return {"predicted_price": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
