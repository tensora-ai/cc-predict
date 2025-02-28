import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router
from app.resources import cleanup_resources, initialize_resources


load_dotenv()

# Prepare app resources to be shared across requests
app_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources at app startup
    initialize_resources()

    yield

    # Clean up resources after app shutdown
    cleanup_resources()


# Initialize the FastAPI app
app = FastAPI(
    title="Tensora Count - Predict Backend", version="1.0.0", lifespan=lifespan
)

# Add base router
app.include_router(router, prefix=os.getenv("API_BASE_URL"))
