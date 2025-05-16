from fastapi import APIRouter, Depends
from typing import Annotated

from app.dependencies import get_project_service
from app.services.project_service import ProjectService
from app.utils.auth_utils import check_api_key

router = APIRouter()


@router.get("/check")
def check_projects(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    key: str = Depends(check_api_key),
) -> dict:
    """An endpoint that checks if all entries in the 'projects' CosmosDB container have the correct format."""
    return {"flaws": project_service.check_projects()}
