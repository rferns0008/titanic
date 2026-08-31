"""
FastAPI application for Titanic ML model serving.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
import pickle
from contextlib import asynccontextmanager
import pandas as pd
from config import settings

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model variable
model = None

def preprocess_prediction_input(features: dict) -> list:
    """
    Preprocess raw prediction input to match training format.
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
        name = features.get("Name", "")
        title_str = features.get("Title")

        if not title_str and name:
            # Extract title using regex (e.g., finding "Miss.", "Mr.", etc.)
            match = re.search(r' ([A-Za-z]+)\.', name)
            if match:
                title_str = match.group(1)
            else:
                title_str = "Unknown"
        elif not title_str:
            title_str = "Unknown"
        
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model on application startup, and cleanup on shutdown."""
    global model
    
    # --- STARTUP LOGIC ---
    try:
        if settings.model_path.exists():
            with open(settings.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✓ Model loaded successfully from {settings.model_path}")
        else:
            logger.error(f"✗ Model file not found at {settings.model_path}")
            logger.error(f"  Available models directory contents:")
            models_dir = settings.model_path.parent
            if models_dir.exists():
                for file in models_dir.iterdir():
                    logger.error(f"    - {file.name}")
                    
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}", exc_info=True)
        raise
    
    yield  # The application runs while yielded
    
    # --- SHUTDOWN LOGIC ---
    logger.info("Application shutting down")

# Initialize FastAPI app once with all configurations
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    debug=settings.debug,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

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
        "model_loaded": model is not None
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
    """
    global model
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Preprocess features
        processed_features = preprocess_prediction_input(features)
        
        # Convert to DataFrame to resolve Scikit-Learn UserWarning
        input_df = pd.DataFrame([processed_features])
            
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
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