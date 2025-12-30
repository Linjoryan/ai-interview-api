from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pickle
import numpy as np
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student performance using logistic regression",
    version="1.0.0"
)

# Load the model
MODEL_PATH = "logistic_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None


# Define input schema
class StudentFeatures(BaseModel):
    """
    Input features for student performance prediction.
    """
    sex: int = Field(..., ge=0, le=1, description="Student's sex (0 or 1)")
    age: int = Field(..., ge=15, le=22, description="Student's age (15-22)")
    Medu: int = Field(..., ge=0, le=4, description="Mother's education (0-4)")
    Fedu: int = Field(..., ge=0, le=4, description="Father's education (0-4)")
    famrel: int = Field(..., ge=1, le=5, description="Quality of family relationships (1-5)")
    freetime: int = Field(..., ge=1, le=5, description="Free time after school (1-5)")
    goout: int = Field(..., ge=1, le=5, description="Going out with friends (1-5)")
    Dalc: int = Field(..., ge=1, le=5, description="Workday alcohol consumption (1-5)")
    Walc: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption (1-5)")
    health: int = Field(..., ge=1, le=5, description="Current health status (1-5)")
    absences: int = Field(..., ge=0, description="Number of school absences")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sex": 0,
                "age": 17,
                "Medu": 3,
                "Fedu": 3,
                "famrel": 4,
                "freetime": 3,
                "goout": 2,
                "Dalc": 1,
                "Walc": 2,
                "health": 4,
                "absences": 4
            }
        }
    
    @validator('sex', 'age', 'Medu', 'Fedu', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', pre=True)
    def validate_numeric(cls, v):
        """Ensure all numeric inputs are valid integers"""
        if v is None:
            raise ValueError("Value cannot be None")
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError("Must be a numeric value")


# Define output schema
class PredictionResponse(BaseModel):
    prediction: str
    prediction_label: int
    confidence_score: float
    probability_pass: float
    probability_fail: float


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Student Performance Prediction API",
        "status": "active" if model is not None else "model not loaded",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: StudentFeatures):
    """
    Predict student performance based on input features.
    
    Returns:
    - prediction: "Pass" or "Fail"
    - prediction_label: 1 for Pass, 0 for Fail
    - confidence_score: Confidence of the prediction (0-1)
    - probability_pass: Probability of passing
    - probability_fail: Probability of failing
    """
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please contact administrator."
        )
    
    try:
        # Prepare input data
        input_data = np.array([[
            features.sex,
            features.age,
            features.Medu,
            features.Fedu,
            features.famrel,
            features.freetime,
            features.goout,
            features.Dalc,
            features.Walc,
            features.health,
            features.absences
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get probability scores
        probabilities = model.predict_proba(input_data)[0]
        
        # Determine which class is which (0=Fail, 1=Pass typically)
        prob_fail = probabilities[0]
        prob_pass = probabilities[1]
        
        # Confidence is the maximum probability
        confidence = max(prob_fail, prob_pass)
        
        # Create response
        response = PredictionResponse(
            prediction="Pass" if prediction == 1 else "Fail",
            prediction_label=int(prediction),
            confidence_score=round(float(confidence), 4),
            probability_pass=round(float(prob_pass), 4),
            probability_fail=round(float(prob_fail), 4)
        )
        
        logger.info(f"Prediction successful: {response.prediction}")
        return response
        
    except AttributeError as e:
        logger.error(f"Model method error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Model does not support required prediction methods"
        )
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )


# Run with: uvicorn filename:app --reload