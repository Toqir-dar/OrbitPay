from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base


class PredictionRecord(Base):

    __tablename__ = "predictions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Input features
    age = Column(Integer, nullable=False)
    monthly_inhand_salary = Column(Float, nullable=False)
    num_credit_card = Column(Integer, nullable=False)
    outstanding_debt = Column(Float, nullable=False)
    credit_utilization_ratio = Column(Float, nullable=False)
    monthly_balance = Column(Float, nullable=False)
    credit_mix = Column(String(20), nullable=False)
    
    # Prediction results
    predicted_score = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)  # Store as JSON
    
    # Metadata
    customer_id = Column(String(100), nullable=True, index=True)
    model_version = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # Engineered features (optional - for debugging/analysis)
    debt_to_income = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<PredictionRecord(id={self.id}, score={self.predicted_score}, confidence={self.confidence:.2f})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "age": self.age,
            "monthly_inhand_salary": self.monthly_inhand_salary,
            "num_credit_card": self.num_credit_card,
            "outstanding_debt": self.outstanding_debt,
            "credit_utilization_ratio": self.credit_utilization_ratio,
            "monthly_balance": self.monthly_balance,
            "credit_mix": self.credit_mix,
            "predicted_score": self.predicted_score,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "customer_id": self.customer_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp
        }