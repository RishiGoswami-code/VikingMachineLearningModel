from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import sys
from typing import Literal
from naive import NaiveBayesModel
from logistics import LogisticRegressionModel
from randomForest import RandomForestModel
from SVM import SVMModel
from k_means import KMeansModel


# --- PATH CORRECTION AND DYNAMIC IMPORT SETUP ---

# 1. Get the directory where the current script (fast_app.py / model.py) is located (e.g., .../Backend)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Add the parent directory to the system path to allow importing local modules
#    The parent directory contains the model files (naive.py, logistics.py, etc.)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)


# --- Configuration ---
app = FastAPI(
    title="Authenticity Ensemble Prediction API",
    description="Predicts document authenticity using a 5-model ensemble (RF, SVM, LR, K-Means, NB)."
)

# Define the path to the 'data' folder, which is sibling to the 'Backend' folder
DATA_FILE_NAME = "data.json" 
# Correct Path: [Parent Dir] / data / data.json
DATA_FILE_PATH = os.path.join(parent_dir, "data", DATA_FILE_NAME) 

# Global variables to hold the trained models
MODELS = {}

# --- Pydantic Data Model for Request Body ---

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
    

# --- Helper Function to Load Data (Uses the corrected path) ---

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

# --- Model Loading and Training Function (Runs on startup) ---

@app.on_event("startup")
async def load_and_train_models():
    """Loads the data and trains all five models once when the API starts."""
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
        MODELS['rf'] = RandomForestModel(training_data, weights)
        MODELS['svm'] = SVMModel(training_data, weights, kernel='linear', C=1.0)
        MODELS['lr'] = LogisticRegressionModel(training_data, weights, learning_rate=0.1, n_iterations=5000)
        MODELS['kmeans'] = KMeansModel(training_data, weights)
        MODELS['nb'] = NaiveBayesModel(training_data, weights)
        
        print("All 5 Models Trained and Ready.")
        
    except Exception as e:
        print(f"FATAL ERROR during model training: {e}")
        MODELS = None
        raise RuntimeError(f"Model Training Failed: {e}")


# --- Ensemble Prediction Logic (Unchanged) ---

def ensemble_predict(test_case):
    """
    Runs the test case through all 5 models and returns the final voted prediction.
    Priority Order: RF > SVM > LR > K-Means > NB
    """
    
    # Get individual model predictions
    rf_prediction = MODELS['rf'].result(test_case)['prediction']
    svm_prediction = MODELS['svm'].result(test_case)['prediction']
    lr_prediction = MODELS['lr'].result(test_case)['prediction']
    kmeans_prediction = MODELS['kmeans'].result(test_case)['prediction']
    nb_prediction = MODELS['nb'].predict(test_case)['prediction']

    # Collect all predictions
    predictions = [rf_prediction, svm_prediction, lr_prediction, kmeans_prediction, nb_prediction]
    genuine_votes = predictions.count('genuine')
    fraud_votes = predictions.count('fraud')

    if genuine_votes > fraud_votes:
        return "genuine"
    elif fraud_votes > genuine_votes:
        return "fraud"
    else:
        # Tie-breaker: defaults to the highest priority RF
        return rf_prediction

# --- FastAPI Endpoint (Unchanged) ---

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
        
        # Map the string result to the requested boolean output
        is_genuine = (final_prediction == 'genuine')

        return {
            "prediction_status": is_genuine,
            "prediction_label": final_prediction.upper()
        }

    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to internal error.")