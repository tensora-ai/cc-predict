from fastapi import APIRouter

router = APIRouter()


@router.get("")
def health_check():
    """Simple healthcheck that returns 200 OK."""
    return {"status": "SUCCESS"}
