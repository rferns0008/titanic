"""
FastAPI application for Titanic ML model serving.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional
import pickle
from pathlib import Path

from config import settings


# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    debug=settings.debug,
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Load model on startup
model = None
preprocessor = None


def preprocess_prediction_input(features: dict) -> list:
    """
    Preprocess raw prediction input to match training format.
    
    Input example:
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
        "Title": "Mr"  # optional, defaults to "Unknown"
    }
    """
    try:
        # Extract and validate basic features
        pclass = int(features.get("Pclass", 3))
        sex_str = str(features.get("Sex", "male")).lower()
        age = float(features.get("Age", 30.0))
        sibsp = int(features.get("SibSp", 0))
        parch = int(features.get("Parch", 0))
        fare = float(features.get("Fare", 0.0))
        embarked_str = str(features.get("Embarked", "S")).upper()
        title_str = str(features.get("Title", "Unknown"))
        
        # Map categorical features
        sex = settings.sex_mapping.get(sex_str)
        if sex is None:
            raise ValueError(f"Invalid Sex value: '{sex_str}'. Must be 'male' or 'female'")
        
        embarked = settings.embarked_mapping.get(embarked_str)
        if embarked is None:
            raise ValueError(f"Invalid Embarked value: '{embarked_str}'. Must be 'S', 'C', or 'Q'")
        
        title = settings.title_mapping.get(title_str, settings.title_mapping.get("Unknown"))
        
        # Calculate derived features
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        
        # Return features in expected order
        return [
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare,
            embarked,
            family_size,
            is_alone,
            title,
        ]
    except (ValueError, KeyError) as e:
        raise ValueError(f"Feature preprocessing failed: {str(e)}")


@app.on_event("startup")
async def load_model():
    """Load the trained model and preprocessor on application startup."""
    global model, preprocessor
    
    try:
        # Load model
        if settings.model_path.exists():
            with open(settings.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✓ Model loaded successfully from {settings.model_path}")
        else:
            logger.error(f"✗ Model file not found at {settings.model_path}")
            logger.error(f"  Looked for: {settings.model_path}")
            logger.error(f"  Available models directory contents:")
            models_dir = settings.model_path.parent
            if models_dir.exists():
                for file in models_dir.iterdir():
                    logger.error(f"    - {file.name}")
        
        # Load preprocessor if it exists
        if settings.preprocessor_path.exists():
            with open(settings.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info(f"✓ Preprocessor loaded from {settings.preprocessor_path}")
        else:
            logger.warning(f"⚠ Preprocessor not found at {settings.preprocessor_path} (optional)")
            
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Application shutting down")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Titanic ML API",
        "version": settings.api_version,
        "status": "running"
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.get("/config", tags=["Info"])
async def get_config():
    """Get API configuration."""
    return {
        "model_name": settings.model_name,
        "model_version": settings.model_version,
        "expected_features": settings.expected_features,
    }


@app.post("/predict", tags=["Predictions"])
async def predict(features: dict):
    """
    Make a prediction using the trained model.
    
    Expected input format with raw categorical values:
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
        "Title": "Mr"
    }
    
    Note: Title is optional (defaults to "Unknown" if not provided)
    """
    global model, preprocessor
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Preprocess features
        processed_features = preprocess_prediction_input(features)
        
        # Make prediction
        prediction = model.predict([processed_features])
        probability = model.predict_proba([processed_features])
        
        return {
            "prediction": int(prediction[0]),
            "prediction_label": "Survived" if prediction[0] == 1 else "Did not survive",
            "probability": {
                "did_not_survive": float(probability[0][0]),
                "survived": float(probability[0][1])
            },
            "confidence": float(max(probability[0])),
            "model_version": settings.model_version
        }
    except ValueError as e:
        logger.warning(f"Validation error in prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )
