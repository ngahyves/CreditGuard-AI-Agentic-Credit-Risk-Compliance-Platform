# src/database/models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class InferenceLog(Base):
    """
    INFERENCE STORE TABLE
    Archives every AI decision for compliance, auditing, and drift monitoring.
    """
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(50), unique=True, index=True)
    client_id = Column(Integer, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # ML Results
    credit_score = Column(Integer)
    verdict = Column(String(20))
    risk_level = Column(String(10))
    probability_of_default = Column(Float)
    strategy_used = Column(String(10))
    
    # AI Narrative
    final_memo = Column(Text)
    # SHAP reasons stored as JSON for future retraining
    top_risk_drivers = Column(JSON)