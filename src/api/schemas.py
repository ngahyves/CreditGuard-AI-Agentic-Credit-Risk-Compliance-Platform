# src/api/schemas.py

import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class LoanApplicationRequest(BaseModel):
    """
    The 'Envelope' for the incoming request.
    Pydantic only checks that we have an ID and a dictionary of features.
    """
    SK_ID_CURR: int = Field(..., example=100002)
    features: Dict[str, Any] = Field(..., description="Key-value pairs of the 85 features")

class AppraisalResponse(BaseModel):
    """
    The structured response sent back to the bank's front-end.
    """
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    client_id: int
    credit_score: int
    verdict: str
    risk_level: str
    top_risk_drivers: List[Dict[str, Any]]
    final_memo: str