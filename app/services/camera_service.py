from typing import Optional, Tuple
from datetime import time

from app.models.project import Camera, CameraConfig, CountingModel, Area
from app.services.project_service import ProjectService


class CameraService:
    """Service for camera-related operations."""

    def __init__(self, project_service: ProjectService = None):
        """Initialize with a project service."""
        self._project_service = project_service or ProjectService()

    def get_camera(self, project_id: str, camera_id: str) -> Optional[Camera]:
        """Get a camera from a project."""
        project = self._project_service.get_project(project_id)
        if not project:
            return None

        for camera in project.cameras:
            if camera.id == camera_id:
                return camera

        return None

    def get_camera_config(
        self, project_id: str, camera_id: str, position_name: str
    ) -> Optional[Tuple[CameraConfig, Area]]:
        """Get camera configuration and area for a specific camera and position."""
        project = self._project_service.get_project(project_id)
        if not project:
            return None

        for area in project.areas:
            for camera_config in area.camera_configs:
                if (
                    camera_config.camera_id == camera_id
                    and camera_config.position.name == position_name
                ):
                    return (camera_config, area)

        return None

    def get_active_model(
        self, project_id: str, camera_id: str, current_time: time
    ) -> CountingModel:
        """Get the active model for a camera at the current time."""
        camera = self.get_camera(project_id, camera_id)
        if not camera:
            return CountingModel.STANDARD  # Default fallback

        return camera.get_active_model(current_time)
