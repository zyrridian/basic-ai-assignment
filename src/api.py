"""
FastAPI Backend for Burnout AI
Serves the trained neural network model via REST API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import numpy as np

from predictor import predict_burnout_risk, load_model_weights

# Initialize FastAPI app
app = FastAPI(
    title="Burnout AI API",
    description="Binary classification API for predicting burnout risk based on lifestyle habits",
    version="1.0.0"
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model weights at startup
W1, W2 = load_model_weights()
print("✓ Model weights loaded successfully!")


# Request/Response models
class PredictionRequest(BaseModel):
    """Request body for burnout prediction"""
    sleep_hours: float = Field(..., ge=0, le=12, description="Hours of sleep per night")
    work_hours: float = Field(..., ge=0, le=24, description="Hours of work/study per day")
    relax_hours: float = Field(..., ge=0, le=12, description="Hours of relaxation per day")
    
    class Config:
        schema_extra = {
            "example": {
                "sleep_hours": 6,
                "work_hours": 10,
                "relax_hours": 2
            }
        }


class PredictionResponse(BaseModel):
    """Response body for burnout prediction"""
    success: bool
    probability: float = Field(..., description="Burnout risk percentage (0-100)")
    risk_level: str = Field(..., description="Risk classification: healthy/caution/warning/danger")
    status_emoji: str
    recommendation: str
    inputs: dict


@app.get("/")
async def root():
    """Root endpoint - serve web interface"""
    web_path = Path(__file__).parent.parent / "web" / "index.html"
    if web_path.exists():
        return FileResponse(web_path)
    return {
        "message": "Burnout AI API",
        "docs": "/docs",
        "prediction_endpoint": "/predict"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": W1 is not None and W2 is not None,
        "model_architecture": {
            "input_size": 3,
            "hidden_size": 4,
            "output_size": 1
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict burnout risk based on lifestyle inputs
    
    Args:
        request: PredictionRequest with sleep, work, and relax hours
    
    Returns:
        PredictionResponse with risk assessment and recommendations
    """
    try:
        # Validate total hours (should be reasonable)
        total_hours = request.sleep_hours + request.work_hours + request.relax_hours
        if total_hours > 24:
            raise HTTPException(
                status_code=400,
                detail=f"Total hours ({total_hours}) exceeds 24 hours per day. Please check your inputs."
            )
        
        # Make prediction
        result = predict_burnout_risk(
            sleep_hours=request.sleep_hours,
            work_hours=request.work_hours,
            relax_hours=request.relax_hours,
            W1=W1,
            W2=W2
        )
        
        return PredictionResponse(
            success=True,
            **result
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/info")
async def api_info():
    """Get information about the API and model"""
    return {
        "model": "Burnout Binary Classification Neural Network",
        "architecture": {
            "layers": [
                {"type": "input", "neurons": 3, "inputs": ["sleep_hours", "work_hours", "relax_hours"]},
                {"type": "hidden", "neurons": 4, "activation": "sigmoid"},
                {"type": "output", "neurons": 1, "activation": "sigmoid", "output": "burnout_probability"}
            ]
        },
        "training": {
            "method": "backpropagation",
            "learning_rate": 0.5,
            "epochs": 20000,
            "algorithm": "gradient_descent"
        },
        "risk_thresholds": {
            "healthy": "< 30%",
            "caution": "30-50%",
            "warning": "50-70%",
            "danger": "> 70%"
        }
    }


# Mount static files for web interface
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting Burnout AI API Server")
    print("=" * 60)
    print("API Documentation: http://localhost:8000/docs")
    print("Web Interface: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
