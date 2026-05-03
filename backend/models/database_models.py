from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    file_path = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    experiments = relationship("ExperimentRun", back_populates="dataset", cascade="all, delete-orphan")

class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    algorithm_type = Column(String) # baseline or enhanced
    parameters = Column(JSON)
    status = Column(String, default="completed")
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="experiments")
    results = relationship("Result", back_populates="run", cascade="all, delete-orphan", uselist=False)

class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("experiment_runs.id"), unique=True)
    cluster_labels = Column(JSON, nullable=True)
    outliers = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    
    silhouette_score = Column(Float, nullable=True)
    davies_bouldin_score = Column(Float, nullable=True)
    num_outliers = Column(Integer, nullable=True)
    variance = Column(Float, nullable=True)
    iterations = Column(Integer, nullable=True)
    history = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("ExperimentRun", back_populates="results")
