import os

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.utils import (
    initialize_model,
    process_project_metadata,
    create_cosmos_db_client,
)

load_dotenv()

# Prepare app resources to be shared across requests
app_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_resources["models"] = {
        "standard": initialize_model(os.environ["STANDARD_MODEL"]),
        "lightshow": initialize_model(os.environ["LIGHTSHOW_MODEL"]),
    }

    app_resources["cosmosdb"] = create_cosmos_db_client("predictions")

    (
        app_resources["masks"],
        app_resources["interpolators"],
        app_resources["gridded_indices"],
        app_resources["model_schedules"],
    ) = process_project_metadata()

    yield

    # Clean up resources after app shutdown
    app_resources.clear()


# Initialize the FastAPI app
app = FastAPI(
    title="Tensora Count - Predict Backend", version="1.0.0", lifespan=lifespan
)
