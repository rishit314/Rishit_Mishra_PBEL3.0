from pydantic import BaseModel, Field
from typing import List, Optional

class DiseaseContext(BaseModel):
    disease_id: str = Field(..., description="Parsed label ID of the condition")
    crop_name: str = Field(..., description="Identified plant species")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model output score")

class PredictResponse(BaseModel):
    disease_id: str
    crop_name: str
    confidence: float
    initial_summary: str

class ChatRequest(BaseModel):
    session_id: str = Field(..., example="farmer-session-99")
    user_message: str = Field(..., example="Can I apply copper sulfate during bloom?")
    disease_context: Optional[DiseaseContext] = Field(
        None, 
        description="Optional explicit context override; otherwise falls back to cached context"
    )