import json
import os
import sys
from typing import Literal

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # ✅ NEW IMPORT FOR CORS

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import all six model classes 
from naive import NaiveBayesModel
from logistics import LogisticRegressionModel
from randomForest import RandomForestModel
from SVM import SVMModel
from k_means import KMeansModel
from CNN import CNNModel 

# --- Configuration ---
app = FastAPI(
    title="Authenticity Ensemble Prediction API",
    description="Predicts document authenticity using a 6-model ensemble (CNN, RF, SVM, LR, K-Means, NB)."
)

# -----------------------------------------------------
# ✅ CORS Middleware Configuration
# This block handles the "OPTIONS 405 Method Not Allowed" error
# by allowing cross-origin requests from the frontend.
# -----------------------------------------------------

# Define the origins (domains/ports) that are allowed to access the API.
# Use ["*"] to allow all origins during development.
origins = [
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allows all headers (like Content-Type)
)

# --- Configuration End ---

# Define the path to the 'data' folder
DATA_FILE_NAME = "data.json" 
DATA_FILE_PATH = os.path.join(current_dir, "data", DATA_FILE_NAME) 

# Global variables to hold the trained models
MODELS = {}

# --- Pydantic Data Models (Unchanged) ---

class PredictionInput(BaseModel):
    """Defines the expected input structure and types for the API request."""
    sig_sim: float
    seal_sim: float
    photo_clarity: float
    text_alignment: float
    font_match: Literal['yes', 'no']
    layout_ok: Literal['yes', 'no']

class PredictionOutput(BaseModel):
    """Defines the structure for the API response."""
    prediction_status: bool
    prediction_label: str
    

# --- Helper Function to Load Data (Unchanged) ---

def load_data(file_path):
    """Loads the training data, weights, and test cases from the specified JSON file."""
    if not os.path.exists(file_path):
        print(f"FATAL ERROR: Data file not found at {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Could not parse JSON data in {file_path}.")
        return None

# --- Model Loading and Training (Runs on startup) ---

@app.on_event("startup")
async def load_and_train_models():
    """Loads the data and trains all six models once when the API starts."""
    global MODELS

    data_container = load_data(DATA_FILE_PATH)
    if data_container is None:
        print("API startup failed: Could not load data.")
        return

    training_data = data_container.get("training_data")
    weights = data_container.get("weights")

    if not all([training_data, weights]):
        print("API startup failed: Missing training_data or weights.")
        return

    try:
        # Train all six models
        MODELS['cnn'] = CNNModel(training_data, weights)
        MODELS['rf'] = RandomForestModel(training_data, weights)
        MODELS['svm'] = SVMModel(training_data, weights, kernel='linear', C=1.0)
        MODELS['lr'] = LogisticRegressionModel(training_data, weights, learning_rate=0.1, n_iterations=5000)
        MODELS['kmeans'] = KMeansModel(training_data, weights)
        MODELS['nb'] = NaiveBayesModel(training_data, weights)
        
        print("All 6 Models Trained and Ready.")
        
    except Exception as e:
        print(f"FATAL ERROR during model training: {e}")
        MODELS = None
        # Raise an error to prevent the server from starting with untrained models
        raise RuntimeError(f"Model Training Failed: {e}")


# --- Ensemble Prediction Logic (Unchanged) ---

def ensemble_predict(test_case):
    """
    Runs the test case through all 6 models and returns the final voted prediction.
    Priority Order (Highest to Lowest): CNN > RF > SVM > LR > K-Means > NB
    """
    
    # Get individual model predictions
    cnn_prediction = MODELS['cnn'].result(test_case)['prediction'] 
    rf_prediction = MODELS['rf'].result(test_case)['prediction']
    svm_prediction = MODELS['svm'].result(test_case)['prediction']
    lr_prediction = MODELS['lr'].result(test_case)['prediction']
    kmeans_prediction = MODELS['kmeans'].result(test_case)['prediction']
    nb_prediction = MODELS['nb'].predict(test_case)['prediction']

    # Collect all predictions (Total 6 votes)
    # Order reflects priority for tie-breaking: CNN(1) > RF(2) > SVM(3) > LR(4) > K-Means(5) > NB(6)
    predictions = [cnn_prediction, rf_prediction, svm_prediction, lr_prediction, kmeans_prediction, nb_prediction]
    genuine_votes = predictions.count('genuine')
    fraud_votes = predictions.count('fraud')

    if genuine_votes > fraud_votes:
        return "genuine"
    elif fraud_votes > genuine_votes:
        return "fraud"
    else:
        # Tie (3-3 split, which is possible with 6 voters).
        # We must defer to the highest priority model's prediction.
        return cnn_prediction # CNN is the highest priority model in the list

# --- FastAPI Endpoint (Unchanged) ---

@app.get("/")
def read_root():
    """Root endpoint for status check and docs link."""
    return {
        "status": "Online (6 Models)",
        "message": "Ensemble Prediction API is running.",
        "docs_url": "/docs",
        "prediction_endpoint": "/predict (POST)"
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Accepts a validated JSON payload and returns the final prediction status.
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models are not loaded or failed to train.")
    
    # Convert Pydantic model back to a dictionary suitable for model input
    test_case = input_data.dict()
    
    try:
        final_prediction = ensemble_predict(test_case)
        
        # Map the string result to the requested boolean output (true for genuine, false for fraud)
        is_genuine = (final_prediction == 'genuine')

        return {
            "prediction_status": is_genuine,
            "prediction_label": final_prediction.upper()
        }

    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to internal error.")