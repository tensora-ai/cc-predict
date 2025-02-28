from fastapi import APIRouter, Depends

from app.routes.check_database import check_projects_implementation
from app.utils import check_api_key

router = APIRouter()


@router.get("/check")
def check_projects(key: str = Depends(check_api_key)) -> dict:
    """An endpoint that checks if all entries in the 'projects' CosmosDB container have the correct format."""
    return check_projects_implementation()
