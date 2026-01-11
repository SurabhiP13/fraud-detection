"""
ML-based Fraud Detection Model
"""
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FraudDetector:
    """Fraud detection using ML models"""
    
    def __init__(self, model_path: str = None):
        """Initialize fraud detector"""
        self.model_path = model_path or "ml_model/fraud_model.joblib"
        self.scaler_path = "ml_model/scaler.joblib"
        
        # Load or create model
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Loaded existing fraud detection model")
        else:
            # Create a simple model for demonstration
            self.model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
            logger.info("Created new fraud detection model")
    
    def _extract_features(self, transaction: dict) -> np.ndarray:
        """Extract features from transaction"""
        # Simple feature extraction
        features = [
            float(transaction.get('amount', 0)),
            # Encode category as numeric (simplified)
            hash(transaction.get('category', '')) % 100,
            # Encode merchant as numeric (simplified)
            hash(transaction.get('merchant', '')) % 100,
            # Add more features as needed
        ]
        return np.array(features).reshape(1, -1)
    
    def predict(self, transaction: dict) -> bool:
        """Predict if transaction is fraudulent"""
        try:
            features = self._extract_features(transaction)
            
            # Simple rule-based detection for demo purposes
            amount = transaction.get('amount', 0)
            
            # Flag transactions with unusual amounts
            if amount > 10000:  # High amount threshold
                logger.info(f"High amount transaction flagged: {amount}")
                return True
            
            if amount < 0:  # Negative amounts
                return True
            
            # Additional heuristics
            merchant = transaction.get('merchant', '').lower()
            suspicious_keywords = ['unknown', 'test', 'suspicious']
            
            if any(keyword in merchant for keyword in suspicious_keywords):
                logger.info(f"Suspicious merchant flagged: {merchant}")
                return True
            
            # Use model if available and trained
            try:
                if hasattr(self.model, 'predict'):
                    prediction = self.model.predict(features)
                    # IsolationForest returns -1 for anomalies, 1 for normal
                    if hasattr(prediction, '__iter__'):
                        is_fraud = prediction[0] == -1
                    else:
                        is_fraud = prediction == -1
                    return bool(is_fraud)
            except Exception as e:
                logger.warning(f"Model prediction failed, using rules: {e}")
            
            # Default to not fraud if no rules triggered
            return False
            
        except Exception as e:
            logger.error(f"Error in fraud prediction: {str(e)}")
            return False
    
    def get_fraud_score(self, transaction: dict) -> float:
        """Get fraud probability score (0-1)"""
        try:
            features = self._extract_features(transaction)
            amount = transaction.get('amount', 0)
            
            # Calculate score based on amount and other factors
            score = 0.0
            
            if amount > 10000:
                score += 0.7
            elif amount > 5000:
                score += 0.4
            elif amount > 1000:
                score += 0.2
            
            # Check merchant
            merchant = transaction.get('merchant', '').lower()
            if any(kw in merchant for kw in ['unknown', 'test', 'suspicious']):
                score += 0.3
            
            # Try model scoring if available
            try:
                if hasattr(self.model, 'score_samples'):
                    model_score = self.model.score_samples(features)
                    # Normalize to 0-1 range
                    score = max(score, abs(float(model_score[0])) / 10)
            except Exception as e:
                logger.debug(f"Model scoring not available: {e}")
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating fraud score: {str(e)}")
            return 0.0
    
    def train(self, X, y=None):
        """Train the fraud detection model"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            logger.info("Model trained and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
