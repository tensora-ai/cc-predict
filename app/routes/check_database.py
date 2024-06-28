from fastapi import HTTPException

from app.utils.database_helper_functions import create_cosmos_db_client


# ------------------------------------------------------------------------------
def check_projects_implementation() -> dict:
    try:
        projects_client = create_cosmos_db_client("projects")
        projects = projects_client.query_items(
            query="SELECT * FROM c", enable_cross_partition_query=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving projects from CosmosDB: {e}.",
        )

    flaws = {}
    for p in projects:
        project_flaws = []
        for key in ["key", "name", "lat", "lon", "cameras", "areas"]:
            if key not in p.keys():
                project_flaws.append(f"Field {key} is missing.")

        # --- Loop through areas ---
        for area in p["areas"].values():
            for key in ["name", "capacity", "lat", "lon"]:
                if key not in area.keys():
                    project_flaws.append(
                        f"Area {area}: field {key} is missing."
                    )

        if len(list(p["cameras"].keys())) == 0:
            project_flaws.append(f"No cameras entered.")

        # --- Loop through camera settings ---
        for camera, camera_settings in p["cameras"].items():
            for key in ["resolution", "position_settings"]:
                if key not in camera_settings.keys():
                    project_flaws.append(
                        f"Camera {camera}: field {key} is missing."
                    )

            resolution_correct = False
            if not isinstance(camera_settings["resolution"], list):
                project_flaws.append(
                    f"Field 'resolution' of camera {camera} needs to be a list."
                )
            else:
                if len(camera_settings["resolution"]) != 2:
                    project_flaws.append(
                        f"Field 'resolution' of camera {camera} needs to have exactly 2 entries."
                    )
                else:
                    resolution_correct = True

            perspective_transformation = any(
                [
                    item in camera_settings.keys()
                    for item in ["sensor_size", "coordinates_3D"]
                ]
            )
            if perspective_transformation:
                if not all(
                    [
                        item in camera_settings.keys()
                        for item in ["sensor_size", "coordinates_3D"]
                    ]
                ):
                    project_flaws.append(
                        f"All of the fields 'sensor_size' and 'coordinates_3D' must be given for camera {camera} if one wants to do perspective transformations."
                    )

            if "sensor_size" in camera_settings.keys():
                if not isinstance(camera_settings["sensor_size"], list):
                    project_flaws.append(
                        f"Field 'sensor_size' of camera {camera} needs to be a list."
                    )
                else:
                    if len(camera_settings["sensor_size"]) != 2:
                        project_flaws.append(
                            f"Field 'sensor_size' of camera {camera} needs to have exactly 2 entries."
                        )

            if "coordinates_3D" in camera_settings.keys():
                if not isinstance(camera_settings["coordinates_3D"], list):
                    project_flaws.append(
                        f"Field 'coordinates_3D' of camera {camera} needs to be a list."
                    )
                else:
                    if len(camera_settings["coordinates_3D"]) != 3:
                        project_flaws.append(
                            f"Field 'coordinates_3D' of camera {camera} needs to have exactly 3 entries."
                        )

            # --- Loop through position settings ---
            for position, position_settings in camera_settings[
                "position_settings"
            ].items():
                if "center_ground_plane" in position_settings.keys():
                    if not isinstance(
                        position_settings["center_ground_plane"], list
                    ):
                        project_flaws.append(
                            f"Field 'center_ground_plane' of camera {camera}, position {position} needs to be a list."
                        )
                    else:
                        if len(position_settings["center_ground_plane"]) != 2:
                            project_flaws.append(
                                f"Field 'center_ground_plane' of camera {camera}, position {position} needs to have exactly 2 entries."
                            )

                if perspective_transformation or any(
                    [
                        item in position_settings.keys()
                        for item in ["center_ground_plane", "focal_length"]
                    ]
                ):
                    if not all(
                        [
                            item in position_settings.keys()
                            for item in [
                                "center_ground_plane",
                                "focal_length",
                            ]
                        ]
                    ):
                        project_flaws.append(
                            f"All of the fields 'center_ground_plane' and 'focal_length' must be given for camera {camera}, position {position} if one wants to do perspective transformations."
                        )
                if "interpolation_settings" in position_settings.keys():
                    if isinstance(
                        position_settings["interpolation_settings"], dict
                    ):
                        project_flaws.append(
                            f"Camera {camera}, position {position}: field 'interpolation_settings' needs to be a dict."
                        )
                    else:
                        for key in ["radius", "p", "threshold"]:
                            if key not in area_metadata.keys():
                                project_flaws.append(
                                    f"Camera {camera}, position {position}: field {key} is missing inside field 'interpolation_settings'."
                                )

                # --- Loop through area metadata ---
                for area, area_metadata in position_settings[
                    "area_metadata"
                ].items():
                    if area not in p["areas"].keys():
                        project_flaws.append(
                            f"Camera {camera}, position {position}, area metadata {area}: Specified area not given in project field 'areas'."
                        )

                    for key in ["interpolate", "edges"]:
                        if key not in area_metadata.keys():
                            project_flaws.append(
                                f"Camera {camera}, position {position}, area metadata {area}: field {key} is missing."
                            )

                    if "edges" in area_metadata.keys():
                        if not isinstance(area_metadata["edges"], list):
                            project_flaws.append(
                                f"Camera {camera}, position {position}, area metadata {area}: field 'edges' needs to be a list of lists."
                            )
                        for edge in area_metadata["edges"]:
                            if not isinstance(edge, list):
                                project_flaws.append(
                                    f"Camera {camera}, position {position}, area metadata {area}: field 'edges' needs to be a list of lists."
                                )
                            else:
                                if len(edge) != 2:
                                    project_flaws.append(
                                        f"Camera {camera}, position {position}, area metadata {area}: every edge needs to have exactly two entries."
                                    )
                                else:
                                    if resolution_correct:
                                        if (
                                            edge[0]
                                            > camera_settings["resolution"][0]
                                            or edge[1]
                                            > camera_settings["resolution"][1]
                                        ):
                                            project_flaws.append(
                                                f"Camera {camera}, position {position}, area metadata {area}: edge {edge} not compatible with given camera resolution."
                                            )

        if len(list(project_flaws)) > 0:
            flaws[p["id"]] = project_flaws

    return {"flaws": flaws}
