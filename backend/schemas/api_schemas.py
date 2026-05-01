from pydantic import BaseModel
from typing import Optional, Any, Dict, List
from datetime import datetime

class DatasetOut(BaseModel):
    id: int
    name: str
    file_path: str
    uploaded_at: datetime

    class Config:
        from_attributes = True

class ExperimentRunCreate(BaseModel):
    dataset_id: int
    algorithm_type: str
    parameters: Dict[str, Any]

class ResultOut(BaseModel):
    id: int
    run_id: int
    cluster_labels: List[int]
    outliers: List[int]
    metrics: Dict[str, float]
    created_at: datetime

    class Config:
        from_attributes = True

class ExperimentRunOut(BaseModel):
    id: int
    dataset_id: int
    algorithm_type: str
    parameters: Dict[str, Any]
    created_at: datetime
    results: Optional[ResultOut] = None

    class Config:
        from_attributes = True
