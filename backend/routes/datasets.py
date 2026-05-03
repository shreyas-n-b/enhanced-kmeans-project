from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
import os
import shutil

from backend.database import get_db
from backend.models.database_models import Dataset
from backend.schemas.api_schemas import DatasetOut

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("/", response_model=List[DatasetOut])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset))
    datasets = result.scalars().all()
    return datasets

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    await db.delete(dataset)
    await db.commit()
    
    # DUMMY: Not deleting the actual file from disk here
    return {"message": "Dataset deleted successfully"}

@router.post("/upload-dataset/")
async def upload_dataset_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    if not (file.filename.endswith('.csv') or file.filename.endswith('.txt')):
        raise HTTPException(status_code=400, detail="Unsupported format. Only .csv and .txt are allowed.")
    
    upload_dir = os.path.join("backend", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        new_dataset = Dataset(name=file.filename, file_path=file_path)
        db.add(new_dataset)
        await db.commit()
        await db.refresh(new_dataset)
        
        return {"file_path": file_path, "dataset_id": new_dataset.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
