"""
FastAPI application for Credit Score Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

from app.config import settings
from app.database import init_db, engine
from app.services.prediction_service import prediction_service
from app.routers import prediction
from app.models.predictions import HealthCheck


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("="*80)
    print(f"Starting {settings.APP_NAME}")
    print("="*80)
    
    # Initialize database
    print("\n[1/2] Initializing database...")
    init_db()
    print("✓ Database initialized")
    
    # Load ML models
    print("\n[2/2] Loading ML models...")
    try:
        prediction_service.load_models()
        print("✓ ML models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        raise
    
    print("\n" + "="*80)
    print("✓ Application startup complete!")
    print("="*80)
    
    yield
    
    # Shutdown
    print("\nShutting down application...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API for predicting customer credit scores using machine learning",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to OrbitPay Credit Score Prediction API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Health check endpoint"""
    
    # Check model status
    model_loaded = prediction_service.is_loaded()
    
    # Check database connection
    try:
        engine.connect()
        db_connected = True
    except:
        db_connected = False
    
    status = "healthy" if (model_loaded and db_connected) else "unhealthy"
    
    if not model_loaded or not db_connected:
        raise HTTPException(
            status_code=503,
            detail={
                "status": status,
                "model_loaded": model_loaded,
                "database_connected": db_connected
            }
        )
    
    return HealthCheck(
        status=status,
        timestamp=datetime.now(),
        model_loaded=model_loaded,
        database_connected=db_connected
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )