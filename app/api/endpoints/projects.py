from fastapi import APIRouter, Depends

from app.repositories.project_repository import ProjectRepository
from app.services.project_service import ProjectService
from app.utils.auth_utils import check_api_key
from app.main import app_resources

router = APIRouter()


@router.get("/check")
def check_projects(key: str = Depends(check_api_key)) -> dict:
    """An endpoint that checks if all entries in the 'projects' CosmosDB container have the correct format."""
    return {"flaws": app_resources["project_service"].check_projects()}
