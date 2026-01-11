"""
FastAPI Backend for Fraud Detection System
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import logging

from .database import get_db, init_db
from .models import Transaction, TransactionCreate, TransactionResponse
from .fraud_detector import FraudDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fraud detector
fraud_detector = FraudDetector()


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    logger.info("Application started successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Fraud Detection API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/transactions/start", response_model=dict)
async def start_transaction_processing(db: Session = Depends(get_db)):
    """Start processing transactions through the fraud detection pipeline"""
    try:
        logger.info("Starting transaction processing")
        # This would trigger the Airflow DAG
        return {
            "status": "processing_started",
            "message": "Transaction processing pipeline initiated"
        }
    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transactions", response_model=TransactionResponse)
async def create_transaction(
    transaction: TransactionCreate,
    db: Session = Depends(get_db)
):
    """Create and process a new transaction"""
    try:
        # Predict fraud
        is_fraud = fraud_detector.predict(transaction.dict())
        
        # Save to database
        db_transaction = Transaction(
            **transaction.dict(),
            is_fraud=is_fraud,
            fraud_score=fraud_detector.get_fraud_score(transaction.dict())
        )
        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)
        
        logger.info(f"Transaction {db_transaction.id} processed - Fraud: {is_fraud}")
        return db_transaction
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error processing transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transactions", response_model=List[TransactionResponse])
async def get_transactions(
    skip: int = 0,
    limit: int = 100,
    fraud_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get all transactions, optionally filtered by fraud status"""
    try:
        query = db.query(Transaction)
        if fraud_only:
            query = query.filter(Transaction.is_fraud == True)
        
        transactions = query.order_by(
            Transaction.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return transactions
    
    except Exception as e:
        logger.error(f"Error fetching transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transactions/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(transaction_id: int, db: Session = Depends(get_db)):
    """Get a specific transaction by ID"""
    transaction = db.query(Transaction).filter(
        Transaction.id == transaction_id
    ).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return transaction


@app.get("/api/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """Get fraud detection statistics"""
    try:
        total_transactions = db.query(Transaction).count()
        fraud_transactions = db.query(Transaction).filter(
            Transaction.is_fraud == True
        ).count()
        
        return {
            "total_transactions": total_transactions,
            "fraud_transactions": fraud_transactions,
            "legitimate_transactions": total_transactions - fraud_transactions,
            "fraud_percentage": (
                (fraud_transactions / total_transactions * 100)
                if total_transactions > 0 else 0
            )
        }
    
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
