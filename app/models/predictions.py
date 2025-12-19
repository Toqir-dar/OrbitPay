from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


class CreditMixEnum(str, Enum):
    """Credit Mix categories"""
    STANDARD = "Standard"
    GOOD = "Good"
    BAD = "Bad"


class CreditScoreEnum(str, Enum):
    """Credit Score categories"""
    POOR = "Poor"
    STANDARD = "Standard"
    GOOD = "Good"


class PredictionInput(BaseModel):
    """
    Input model for credit score prediction
    All features required by the ML model
    """
    age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    monthly_inhand_salary: float = Field(..., ge=0, description="Monthly in-hand salary")
    num_credit_card: int = Field(..., ge=0, le=20, description="Number of credit cards (0-20)")
    outstanding_debt: float = Field(..., ge=0, description="Outstanding debt amount")
    credit_utilization_ratio: float = Field(..., ge=0, le=100, description="Credit utilization ratio (0-100%)")
    monthly_balance: float = Field(..., description="Monthly balance")
    credit_mix: CreditMixEnum = Field(..., description="Credit mix category")
    
    # Optional metadata
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "monthly_inhand_salary": 5000.0,
                "num_credit_card": 3,
                "outstanding_debt": 2500.0,
                "credit_utilization_ratio": 35.5,
                "monthly_balance": 1500.0,
                "credit_mix": "Good",
                "customer_id": "CUST001"
            }
        }
    
    @validator('credit_utilization_ratio')
    def validate_utilization(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Credit utilization ratio must be between 0 and 100')
        return v
    
    @validator('monthly_inhand_salary', 'outstanding_debt')
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v


class PredictionOutput(BaseModel):

    prediction_id: int = Field(..., description="Unique prediction ID from database")
    predicted_score: CreditScoreEnum = Field(..., description="Predicted credit score category")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    probabilities: dict = Field(..., description="Probability for each class")
    
    # Input echo
    input_data: PredictionInput
    
    # Metadata
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="ML model version used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": 1,
                "predicted_score": "Good",
                "confidence": 0.87,
                "probabilities": {
                    "Poor": 0.05,
                    "Standard": 0.08,
                    "Good": 0.87
                },
                "input_data": {
                    "age": 35,
                    "monthly_inhand_salary": 5000.0,
                    "num_credit_card": 3,
                    "outstanding_debt": 2500.0,
                    "credit_utilization_ratio": 35.5,
                    "monthly_balance": 1500.0,
                    "credit_mix": "Good",
                    "customer_id": "CUST001"
                },
                "timestamp": "2025-01-15T10:30:00",
                "model_version": "1.0.0"
            }
        }


class BatchPredictionInput(BaseModel):
    """Input model for batch predictions"""
    predictions: list[PredictionInput] = Field(..., max_length=100)
    
    @validator('predictions')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError('At least one prediction input required')
        if len(v) > 100:
            raise ValueError('Maximum 100 predictions per batch')
        return v


class BatchPredictionOutput(BaseModel):
    """Output model for batch predictions"""
    results: list[PredictionOutput]
    total_predictions: int
    successful: int
    failed: int = 0


class PredictionHistory(BaseModel):
    """Model for prediction history retrieval"""
    prediction_id: int
    predicted_score: str
    confidence: float
    customer_id: Optional[str]
    timestamp: datetime
    
    class Config:
        from_attributes = True


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    model_loaded: bool
    database_connected: bool