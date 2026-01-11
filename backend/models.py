"""
Database models for fraud detection system
"""
from sqlalchemy import Column, Integer, Float, Boolean, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

Base = declarative_base()


class Transaction(Base):
    """Transaction database model"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    merchant = Column(String, nullable=False)
    category = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    location = Column(String, nullable=True)
    user_id = Column(String, nullable=False)
    
    # Fraud detection fields
    is_fraud = Column(Boolean, default=False)
    fraud_score = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TransactionCreate(BaseModel):
    """Pydantic model for creating transactions"""
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant: str = Field(..., min_length=1, description="Merchant name")
    category: str = Field(..., description="Transaction category")
    location: Optional[str] = Field(None, description="Transaction location")
    user_id: str = Field(..., description="User ID")
    
    class Config:
        schema_extra = {
            "example": {
                "amount": 1250.50,
                "merchant": "Online Store XYZ",
                "category": "shopping",
                "location": "New York, USA",
                "user_id": "user_123"
            }
        }


class TransactionResponse(BaseModel):
    """Pydantic model for transaction responses"""
    id: int
    amount: float
    merchant: str
    category: str
    timestamp: datetime
    location: Optional[str]
    user_id: str
    is_fraud: bool
    fraud_score: float
    created_at: datetime
    
    class Config:
        from_attributes = True
