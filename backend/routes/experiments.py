from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import uuid
import asyncio
from typing import Optional

from backend.services.ml_service import run_experiment

router = APIRouter(prefix="/experiments", tags=["Experiments"])

class ExperimentInput(BaseModel):
    file_path: str
    use_enhanced: bool

class ExperimentResponse(BaseModel):
    run_id: str
    status: str

class ExperimentResultOut(BaseModel):
    run_id: str
    algorithm: str
    silhouette_score: float
    davies_bouldin_score: float
    num_outliers: Optional[int] = None

# In-memory store
results_store = {}

@router.post("/run", response_model=ExperimentResponse)
async def run_experiment_endpoint(payload: ExperimentInput):
    # Check if file exists
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        result = await asyncio.to_thread(run_experiment, payload.file_path, payload.use_enhanced)
        run_id = str(uuid.uuid4())
        
        # Add run_id to result for easy retrieval
        result["run_id"] = run_id
        results_store[run_id] = result
        
        return {"run_id": run_id, "status": "completed"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{run_id}", response_model=ExperimentResultOut)
async def get_experiment_result(run_id: str):
    if run_id not in results_store:
        raise HTTPException(status_code=404, detail="Experiment run not found")
    
    return results_store[run_id]
