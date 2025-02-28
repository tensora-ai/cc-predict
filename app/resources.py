import os
from typing import Optional

from app.models.resources import AppResources, CrowdCountingModels
from app.utils.model_prediction.make_prediction import initialize_model
from app.utils.startup import process_project_metadata


# Global variable to store shared app resources
_app_resources: Optional[AppResources] = None


def initialize_resources(Depends):
    """Initialize all application resources"""
    global _app_resources

    # Initialize models
    models = CrowdCountingModels(
        standard=initialize_model(os.environ["STANDARD_MODEL"]),
        lightshow=initialize_model(os.environ["LIGHTSHOW_MODEL"]),
    )

    # Process project metadata
    masks, interpolators, gridded_indices, model_schedules = process_project_metadata()

    # Create the AppResources instance
    _app_resources = AppResources(
        models=models,
        masks=masks,
        interpolators=interpolators,
        gridded_indices=gridded_indices,
        model_schedules=model_schedules,
    )

    return _app_resources


def cleanup_resources():
    """Clean up application resources"""
    global _app_resources
    _app_resources = None


# Dependency function to access app resources
async def get_app_resources() -> AppResources:
    if _app_resources is None:
        raise RuntimeError("Application resources not initialized")
    return _app_resources
