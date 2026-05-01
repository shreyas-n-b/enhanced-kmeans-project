from fastapi import FastAPI
from backend.database import engine, Base
from backend.routes.datasets import router as datasets_router
from backend.routes.experiments import router as experiments_router

app = FastAPI(title="Enhanced K-Means API", version="1.0.0")


# Setup database tables on startup
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Include routers
app.include_router(datasets_router)
app.include_router(experiments_router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Enhanced K-Means Clustering Backend!"}