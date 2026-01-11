"""
Database configuration and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
import os

# Database URL from environment or default to SQLite
# In production, always set DATABASE_URL environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Use SQLite for local development if DATABASE_URL not set
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./fraud_detection.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # PostgreSQL or other database
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
