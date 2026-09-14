import copy
import math

import numpy as np
import open3d as o3d


def preprocess_point_cloud(
    cloud: o3d.geometry.PointCloud,
    control_points: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    Translate and rotate a point cloud using three control points.

    Control points:
        P0 - reference point (new origin)
        P1 - start of reference direction
        P2 - end of reference direction

    The cloud is:
        1. translated so that P0 becomes (0, 0, 0)
        2. rotated around Z so that P1->P2 aligns with +Y.
    """

    if control_points.shape != (3, 3):
        raise ValueError("Expected exactly three control points.")

    cloud = copy.deepcopy(cloud)

    # ---------------------------------------------------------
    # Translate
    # ---------------------------------------------------------

    p0 = control_points[0]
    p1 = control_points[1] - p0
    p2 = control_points[2] - p0

    cloud.translate(-p0)

    # ---------------------------------------------------------
    # Rotation around Z
    # ---------------------------------------------------------

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Rotate P1->P2 onto +Y (angle of +Y is +pi/2).
    theta = (math.pi / 2.0) - math.atan2(dy, dx)

    rotation_matrix = cloud.get_rotation_matrix_from_xyz((0.0, 0.0, theta))

    cloud.rotate(rotation_matrix, center=(0, 0, 0))

    return cloud
