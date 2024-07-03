import numpy as np


# ------------------------------------------------------------------------------
class PerspectiveTransformer:
    def __init__(
        self,
        focal_length: float,
        cam_position: list[float],
        cam_center: tuple[float, float],
    ) -> None:
        """
        focal_length: Focal length of the camera in m.
        camera_position: 3D vector with the camera position in external coordinates in m. The coordinate system is assumed to be left-handed with the x-axis pointing east, the y-axis pointing up, and the z-axis pointing north.
        cam_center: 2D position of the camera center in the y=0 plane in external coordinates in m. The coordinate system is assumed to be left-handed with the x-axis pointing east and the z-axis pointing north.
        """
        if len(cam_position) != 3:
            raise ValueError("Camera position must be a 3D vector.")

        self.focal_length = focal_length

        cam_center_plane = np.array(
            [cam_center[0], 0.0, cam_center[1]], dtype="double"
        )
        self.__calculate_rotation_and_translation__(
            np.array(cam_position, dtype="double"), cam_center_plane
        )

    # --------------------------------------------------------------------------
    def transform_to_ground_plane(
        self, camera_points: tuple[float, float] | list[tuple[float, float]]
    ) -> tuple[float, float] | list[tuple[float, float]]:
        """
        Takes a point (or a list of them) in the camera plain and transforms it to the y=0 plane in external coordinates.
        """
        input_is_tuple = isinstance(camera_points, tuple)
        input_points = [camera_points] if input_is_tuple else camera_points
        result = []

        for point in input_points:
            rho, tau = self.__calculate_solver_system__(point)
            ground_coordinates = np.linalg.solve(rho, tau)
            result.append((ground_coordinates[0], ground_coordinates[1]))

        return result[0] if input_is_tuple else result

    # --------------------------------------------------------------------------
    def __calculate_rotation_and_translation__(
        self, cam_position: np.array, cam_center_plane: np.array
    ) -> None:
        """
        Calculates the rotation matrix from external coordinates to camera coordinates and the translation vector in camera coordinates. Both are stored in the class as attributes.
        """
        # Camera direction in camera coordinates
        cam_dir_int = np.array([0.0, 0.0, 1.0], dtype="double")

        # Calculate direction in which the camera is pointing in external coordinates
        cam_dir_ext = cam_center_plane - cam_position
        cam_dir_ext /= np.linalg.norm(cam_dir_ext)

        # Calculate axis and angle of rotation from external coordinates to camera coordinates
        rot_axis = np.cross(cam_dir_ext, cam_dir_int)
        rot_axis /= np.linalg.norm(rot_axis)
        rot_angle = np.arccos(cam_dir_ext @ cam_dir_int)

        # Calculate rotation matrix with Rodrigues formula
        self.rot_mat = self.__calculate_rotation_matrix__(rot_axis, rot_angle)

        # Translation vector in camera coordinates
        self.transl_vec = self.rot_mat @ (-cam_position)

    # --------------------------------------------------------------------------
    def __calculate_rotation_matrix__(
        self, axis: np.array, theta: float
    ) -> np.array:
        """Calculates the rotation matrix for a given axis and angle using the Rodrigues formula."""
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )

        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

    # --------------------------------------------------------------------------
    def __calculate_solver_system__(
        self, camera_point: tuple[float, float]
    ) -> tuple[np.array, np.array]:
        """
        Calculates the solver system for the transformation of a point in the camera plane to the y=0 plane as
            rho * ground_coordinates = tau,
        where rho is a 2x2 matrix and tau is a 2x1 vector. The function returns both rho and tau.
        """
        rho = np.array(
            [
                [
                    self.rot_mat[0, 0]
                    - camera_point[0] / self.focal_length * self.rot_mat[2, 0],
                    self.rot_mat[0, 2]
                    - camera_point[0] / self.focal_length * self.rot_mat[2, 2],
                ],
                [
                    self.rot_mat[1, 0]
                    - camera_point[1] / self.focal_length * self.rot_mat[2, 0],
                    self.rot_mat[1, 2]
                    - camera_point[1] / self.focal_length * self.rot_mat[2, 2],
                ],
            ]
        )

        tau = np.array(
            [
                camera_point[0] / self.focal_length * self.transl_vec[2]
                - self.transl_vec[0],
                camera_point[1] / self.focal_length * self.transl_vec[2]
                - self.transl_vec[1],
            ]
        )

        return rho, tau
