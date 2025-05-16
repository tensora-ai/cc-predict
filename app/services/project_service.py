from typing import List, Optional
from app.models.project import Project
from app.repositories.project_repository import ProjectRepository


class ProjectService:
    """Service for loading projects from repository as Pydantic models."""

    def __init__(self, repository: ProjectRepository = None):
        """Initialize the service with a repository."""
        self._repository = repository or ProjectRepository()

    def get_all_projects(self) -> List[Project]:
        """Get all projects as Pydantic models."""
        projects_data = self._repository.get_all_projects_data()

        projects = []
        for project_data in projects_data:
            try:
                # Direct conversion from JSON to Pydantic
                project = Project.model_validate(project_data)
                projects.append(project)
            except Exception as e:
                print(f"Error parsing project {project_data.get('id', 'unknown')}: {e}")

        return projects

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a single project by ID as a Pydantic model."""
        project_data = self._repository.get_project_data_by_id(project_id)
        if not project_data:
            return None

        try:
            # Direct conversion from JSON to Pydantic
            return Project.model_validate(project_data)
        except Exception as e:
            print(f"Error parsing project {project_id}: {e}")
            return None

    def check_projects(self) -> dict:
        """
        Validates all projects in the database against the expected schema.

        This method checks:
        - Required fields on all projects
        - Camera configuration validity
        - Proper perspective transformation parameters
        - Valid masking configurations

        Returns:
            Dict with project IDs as keys and lists of flaws as values
        """
        flaws = {}

        # Load all projects and validate them
        try:
            projects = self.get_all_projects()

            for project in projects:
                project_flaws = []

                # Check required fields
                if not project.id:
                    project_flaws.append("Field id is missing.")
                if not project.name:
                    project_flaws.append("Field name is missing.")

                # Check cameras
                if not project.cameras:
                    project_flaws.append("No cameras entered.")

                # Check each camera
                for camera in project.cameras:
                    if not camera.resolution:
                        project_flaws.append(
                            f"Camera {camera.id}: field resolution is missing."
                        )

                    # Check perspective transformation requirements
                    perspective_transformation = (
                        camera.sensor_size is not None
                        or camera.coordinates_3d is not None
                    )
                    if perspective_transformation:
                        if not camera.sensor_size or not camera.coordinates_3d:
                            project_flaws.append(
                                f"All of the fields 'sensor_size' and 'coordinates_3d' must be given for camera {camera.id}"
                            )

                    if camera.sensor_size and len(camera.sensor_size) != 2:
                        project_flaws.append(
                            f"Field 'sensor_size' of camera {camera.id} needs to have exactly 2 entries."
                        )

                    if camera.coordinates_3d and len(camera.coordinates_3d) != 3:
                        project_flaws.append(
                            f"Field 'coordinates_3d' of camera {camera.id} needs to have exactly 3 entries."
                        )

                # Check areas and camera configurations
                for area in project.areas:
                    for camera_config in area.camera_configs:
                        position = camera_config.position
                        camera_id = camera_config.camera_id

                        # Check position requirements
                        if not position.name:
                            project_flaws.append(
                                f"Camera {camera_id}, position: name is missing."
                            )

                        # Check for perspective transformation requirements
                        has_perspective_params = (
                            position.center_ground_plane is not None
                            or position.focal_length is not None
                        )

                        # If camera has perspective info or position has some perspective params
                        for camera in project.cameras:
                            if camera.id == camera_id:
                                camera_has_perspective = (
                                    camera.sensor_size is not None
                                    or camera.coordinates_3d is not None
                                )

                                if camera_has_perspective or has_perspective_params:
                                    if (
                                        not position.center_ground_plane
                                        or not position.focal_length
                                    ):
                                        project_flaws.append(
                                            f"All fields 'center_ground_plane' and 'focal_length' must be given "
                                            f"for camera {camera_id}, position {position.name} for perspective transformation."
                                        )

                        # Check masking configuration
                        if (
                            camera_config.enable_masking
                            and not camera_config.masking_config
                        ):
                            project_flaws.append(
                                f"Camera {camera_id}, position {position.name}: masking is enabled but no masking_config is defined."
                            )

                        # Check masking edges
                        if (
                            camera_config.masking_config
                            and len(camera_config.masking_config.edges) < 3
                        ):
                            project_flaws.append(
                                f"Camera {camera_id}, position {position.name}: masking requires at least 3 points."
                            )

                # Add flaws to result if any were found
                if project_flaws:
                    flaws[project.id] = project_flaws

        except Exception as e:
            return {"error": f"An error occurred while checking projects: {str(e)}"}

        return {"flaws": flaws}
