import os

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.routes import router

# Load environment variables
load_dotenv()

# Build the FastAPI app
app = FastAPI(title="Tensora Count - Predict Backend", version="1.0.0")

app.include_router(router, prefix=os.getenv("API_BASE_URL", "/api/v1"))
