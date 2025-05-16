from app.utils.database_helper_functions import create_cosmos_db_client
from app.utils.startup.create_masks import create_masks
from app.utils.startup.selective_idw_interpolator import (
    create_interpolators,
)
from app.utils.startup.perspective.transformed_density_helper_functions import (
    calculate_gridded_indices,
)
from app.models.models import ModelSchedule


# ------------------------------------------------------------------------------
def process_project_metadata() -> tuple[dict, dict, dict, dict]:
    """Creates masks and gridded indices for all projects defined in the corresponding CosmosDB container."""
    projects_client = create_cosmos_db_client("projects")
    projects = projects_client.query_items(
        query="SELECT * FROM c", enable_cross_partition_query=True
    )

    masks = {}
    interpolators = {}
    gridded_indices = {}
    model_schedules = {}
    for p in projects:
        masks[p["id"]] = create_masks(p["cameras"])

        interpolators[p["id"]] = create_interpolators(p["cameras"])

        gridded_indices[p["id"]] = calculate_gridded_indices(p["cameras"])

        model_schedules[p["id"]] = {
            cam_id: ModelSchedule.from_cosmosdb_entry(data["model_schedule"])
            for cam_id, data in p["cameras"].items()
            if "model_schedule" in data.keys()
        }

    return masks, interpolators, gridded_indices, model_schedules
