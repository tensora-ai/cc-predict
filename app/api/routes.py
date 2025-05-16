from fastapi import APIRouter
from app.api.endpoints import health, projects, predictions

# Create the main router
router = APIRouter()

# Include the routers
router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(projects.router, prefix="/projects", tags=["projects"])
router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])


# Add basic root endpoint
@router.get("/")
async def root():
    return {"message": "Welcome to the Tensora Count Predict API"}
