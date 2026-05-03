from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import os
import asyncio
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models.database_models import Dataset, ExperimentRun, Result
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

@router.post("/run", response_model=ExperimentResponse)
async def run_experiment_endpoint(payload: ExperimentInput, db: AsyncSession = Depends(get_db)):
    # Check if file exists
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    # Get dataset by file_path
    result_dataset = await db.execute(select(Dataset).where(Dataset.file_path == payload.file_path))
    dataset = result_dataset.scalars().first()
    
    if not dataset:
        dataset = Dataset(name=os.path.basename(payload.file_path), file_path=payload.file_path)
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        
    try:
        result_dict = await asyncio.to_thread(run_experiment, payload.file_path, payload.use_enhanced)
        
        new_run = ExperimentRun(
            dataset_id=dataset.id,
            algorithm_type=result_dict["algorithm"],
            parameters={"use_enhanced": payload.use_enhanced}
        )
        db.add(new_run)
        await db.commit()
        await db.refresh(new_run)
        
        outliers_data = {"num_outliers": result_dict.get("num_outliers")} if "num_outliers" in result_dict else None
        
        new_result = Result(
            run_id=new_run.id,
            metrics={
                "silhouette_score": result_dict["silhouette_score"],
                "davies_bouldin_score": result_dict["davies_bouldin_score"]
            },
            outliers=outliers_data
        )
        db.add(new_result)
        await db.commit()
        
        return {"run_id": str(new_run.id), "status": "completed"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{run_id}", response_model=ExperimentResultOut)
async def get_experiment_result(run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        run_id_int = int(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id format")
        
    result_run = await db.execute(
        select(ExperimentRun)
        .options(selectinload(ExperimentRun.results))
        .where(ExperimentRun.id == run_id_int)
    )
    run = result_run.scalar_one_or_none()
    
    if not run or not run.results:
        raise HTTPException(status_code=404, detail="Experiment run not found")
        
    res = run.results
    num_outliers = res.outliers.get("num_outliers") if res.outliers else None
    
    return {
        "run_id": str(run.id),
        "algorithm": run.algorithm_type,
        "silhouette_score": res.metrics.get("silhouette_score", 0.0),
        "davies_bouldin_score": res.metrics.get("davies_bouldin_score", 0.0),
        "num_outliers": num_outliers
    }
