"""
API endpoints for credit score predictions
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.database import get_db
from app.models.predictions import (
    PredictionInput, 
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    PredictionHistory
)
from app.models.database_models import PredictionRecord
from app.services.prediction_service import prediction_service
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionOutput)
async def predict_credit_score(
    input_data: PredictionInput,
    db: Session = Depends(get_db)
):

    try:
        
        input_dict = {
            'age': input_data.age,
            'monthly_inhand_salary': input_data.monthly_inhand_salary,
            'num_credit_card': input_data.num_credit_card,
            'outstanding_debt': input_data.outstanding_debt,
            'credit_utilization_ratio': input_data.credit_utilization_ratio,
            'monthly_balance': input_data.monthly_balance,
            'credit_mix': input_data.credit_mix.value
        }
        
        
        predicted_label, confidence, probabilities = prediction_service.predict(input_dict)
        
        
        db_record = PredictionRecord(
            age=input_data.age,
            monthly_inhand_salary=input_data.monthly_inhand_salary,
            num_credit_card=input_data.num_credit_card,
            outstanding_debt=input_data.outstanding_debt,
            credit_utilization_ratio=input_data.credit_utilization_ratio,
            monthly_balance=input_data.monthly_balance,
            credit_mix=input_data.credit_mix.value,
            predicted_score=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            customer_id=input_data.customer_id,
            model_version=prediction_service.model_version,
            debt_to_income=input_data.outstanding_debt / (input_data.monthly_inhand_salary + 1)
        )
        
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        
        return PredictionOutput(
            prediction_id=db_record.id,
            predicted_score=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            input_data=input_data,
            timestamp=db_record.timestamp,
            model_version=prediction_service.model_version
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")




@router.get("/predictions/{prediction_id}", response_model=PredictionOutput)
async def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific prediction by ID"""
    record = db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Reconstruct PredictionInput
    input_data = PredictionInput(
        age=record.age,
        monthly_inhand_salary=record.monthly_inhand_salary,
        num_credit_card=record.num_credit_card,
        outstanding_debt=record.outstanding_debt,
        credit_utilization_ratio=record.credit_utilization_ratio,
        monthly_balance=record.monthly_balance,
        credit_mix=record.credit_mix,
        customer_id=record.customer_id
    )
    
    return PredictionOutput(
        prediction_id=record.id,
        predicted_score=record.predicted_score,
        confidence=record.confidence,
        probabilities=record.probabilities,
        input_data=input_data,
        timestamp=record.timestamp,
        model_version=record.model_version
    )


@router.get("/predictions/customer/{customer_id}", response_model=List[PredictionHistory])
async def get_customer_predictions(
    customer_id: str,
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get prediction history for a specific customer"""
    records = db.query(PredictionRecord)\
        .filter(PredictionRecord.customer_id == customer_id)\
        .order_by(PredictionRecord.timestamp.desc())\
        .limit(limit)\
        .all()
    
    return [
        PredictionHistory(
            prediction_id=r.id,
            predicted_score=r.predicted_score,
            confidence=r.confidence,
            customer_id=r.customer_id,
            timestamp=r.timestamp
        )
        for r in records
    ]


@router.get("/predictions", response_model=List[PredictionHistory])
async def get_all_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    score_filter: Optional[str] = Query(None, description="Filter by score: Poor, Standard, or Good"),
    db: Session = Depends(get_db)
):
    """Get paginated list of all predictions with optional filtering"""
    query = db.query(PredictionRecord)
    
    if score_filter:
        query = query.filter(PredictionRecord.predicted_score == score_filter)
    
    records = query.order_by(PredictionRecord.timestamp.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return [
        PredictionHistory(
            prediction_id=r.id,
            predicted_score=r.predicted_score,
            confidence=r.confidence,
            customer_id=r.customer_id,
            timestamp=r.timestamp
        )
        for r in records
    ]


@router.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """Delete a specific prediction"""
    record = db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    db.delete(record)
    db.commit()
    
    return {"message": f"Prediction {prediction_id} deleted successfully"}