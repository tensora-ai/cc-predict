from fastapi import HTTPException
import os


def check_api_key(key: str):
    if not key or key != os.environ["API_KEY"]:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key
