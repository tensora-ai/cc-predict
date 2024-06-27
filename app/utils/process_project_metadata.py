from app.utils.database_helper_functions import create_cosmos_db_client
from app.utils.predict.predict_helper_functions import create_masks
from app.utils.predict.selective_idw_interpolator import create_interpolators
from app.utils.perspective.transformed_density_helper_functions import (
    calculate_gridded_indices,
)


def process_project_metadata() -> tuple[dict, dict]:
    """Creates masks and gridded indices for all projects defined in the corresponding CosmosDB container."""
    projects_client = create_cosmos_db_client("projects")
    projects = projects_client.query_items(
        query="SELECT * FROM c", enable_cross_partition_query=True
    )

    masks = {}
    interpolators = {}
    gridded_indices = {}
    for p in projects:
        masks[p["id"]] = create_masks(p["cameras"])
        interpolators[p["id"]] = create_interpolators(p["cameras"])
        gridded_indices[p["id"]] = calculate_gridded_indices(p["cameras"])

    print(interpolators)

    return masks, interpolators, gridded_indices
