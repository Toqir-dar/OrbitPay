import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from app.config import settings
from app.utils.feature_engineering import (
    create_features, 
    get_categorical_columns, 
    get_numerical_columns
)


class PredictionService:
   
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.target_mapping = None
        self.reverse_mapping = None
        self.model_version = settings.APP_VERSION
        
    def load_models(self):
        
        try:
            print("Loading ML models...")
            
            
            self.model = joblib.load(settings.BEST_MODEL_PATH)
            print(f"✓ Model loaded: {settings.BEST_MODEL_PATH}")
            
            # Load scaler
            self.scaler = joblib.load(settings.SCALER_PATH)
            print(f"✓ Scaler loaded: {settings.SCALER_PATH}")
            
            # Load encoder
            self.encoder = joblib.load(settings.ENCODER_PATH)
            print(f"✓ Encoder loaded: {settings.ENCODER_PATH}")
            
            # Load target mapping
            self.target_mapping = joblib.load(settings.TARGET_MAPPING_PATH)
            self.reverse_mapping = {v: k for k, v in self.target_mapping.items()}
            print(f"✓ Target mapping loaded: {settings.TARGET_MAPPING_PATH}")
            
            print("✓ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def preprocess_input(self, input_dict: dict) -> np.ndarray:
        
       
        df = create_features(input_dict)
        
        
        cat_cols = get_categorical_columns()
        num_cols = get_numerical_columns(df, cat_cols)
        
        
        encoded = self.encoder.transform(df[cat_cols])
        
        
        X_combined = np.concatenate([df[num_cols].values, encoded], axis=1)
        
        
        X_scaled = self.scaler.transform(X_combined)
        
        return X_scaled
    
    def predict(self, input_dict: dict) -> Tuple[str, float, Dict[str, float]]:
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_models() first.")
        
        
        X_scaled = self.preprocess_input(input_dict)
        
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        
        predicted_label = self.reverse_mapping[prediction]
        confidence = float(probabilities[prediction])
        
        
        prob_dict = {
            self.reverse_mapping[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return predicted_label, confidence, prob_dict
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return all([
            self.model is not None,
            self.scaler is not None,
            self.encoder is not None,
            self.target_mapping is not None
        ])


# Global instance
prediction_service = PredictionService()