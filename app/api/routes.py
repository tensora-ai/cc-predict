from fastapi import APIRouter
from app.api.endpoints import projects, predictions
from app.models.response import HealthCheckResponse

# Create the main router
router = APIRouter()

# Include the profiles router
# router.include_router(projects.router, prefix="/projects", tags=["projects"])


# Add basic root endpoint
@router.get("/")
async def root():
    return {"message": "Welcome to the Profiles API"}


# Add health hcheck endpoint
@router.get("/health")
def healthcheck() -> HealthCheckResponse:
    """Simple healthcheck that returns 200 OK."""
    return HealthCheckResponse(status="HEALTHY")
