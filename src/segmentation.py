from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class SegmentationResult:
    floor: o3d.geometry.PointCloud
    walls: o3d.geometry.PointCloud
    ceiling: o3d.geometry.PointCloud
    cropped: o3d.geometry.PointCloud


def segment_point_cloud(
    cloud: o3d.geometry.PointCloud,
    normal_radius: float = 0.20,
    normal_max_nn: int = 30,
    floor_angle: float = 15.0,
    wall_angle: float = 15.0,
    height_tolerance: float = 0.05,
) -> SegmentationResult:
    """
    Segment floor, walls, ceiling and everything above the wall top.

    Assumptions:
        - Z axis points upwards.
        - Floor and ceiling are horizontal.
        - Walls are vertical.
    """

    cloud = cloud.voxel_down_sample(0.02)

    _estimate_normals(
        cloud,
        radius=normal_radius,
        max_nn=normal_max_nn,
    )

    points = np.asarray(cloud.points)

    horizontal_mask = _detect_horizontal_surfaces(
        cloud,
        floor_angle,
    )

    wall_mask = _detect_walls(
        cloud,
        wall_angle,
    )

    floor_mask, ceiling_mask = _split_floor_ceiling(
        points,
        horizontal_mask,
        height_tolerance,
    )

    floor_cloud = cloud.select_by_index(
        np.where(floor_mask)[0]
    )

    wall_cloud = cloud.select_by_index(
    np.where(wall_mask)[0]
)

    wall_cloud = _remove_floating_points(wall_cloud)

    ceiling_cloud = cloud.select_by_index(
        np.where(ceiling_mask)[0]
    )

    cropped_cloud = _crop_above_walls(
        cloud,
        wall_cloud,
    )

    return SegmentationResult(
        floor=floor_cloud,
        walls=wall_cloud,
        ceiling=ceiling_cloud,
        cropped=cropped_cloud,
    )


def _estimate_normals(
    cloud: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int,
) -> None:

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )

    cloud.orient_normals_consistent_tangent_plane(20)


def _detect_horizontal_surfaces(
    cloud: o3d.geometry.PointCloud,
    angle: float,
) -> np.ndarray:

    normals = np.asarray(cloud.normals)

    z_axis = np.array([0.0, 0.0, 1.0])

    similarity = np.abs(normals @ z_axis)

    threshold = np.cos(np.deg2rad(angle))

    return similarity > threshold


def _detect_walls(
    cloud: o3d.geometry.PointCloud,
    angle: float,
) -> np.ndarray:

    normals = np.asarray(cloud.normals)

    z_axis = np.array([0.0, 0.0, 1.0])

    similarity = np.abs(normals @ z_axis)

    threshold = np.cos(np.deg2rad(90.0 - angle))

    return similarity < threshold

def _remove_floating_points(
    cloud: o3d.geometry.PointCloud,
    nb_neighbors: int = 30,
    std_ratio: float = 1.5,
) -> o3d.geometry.PointCloud:
    """
    Remove isolated floating points while preserving continuous objects.
    """

    filtered_cloud, _ = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    return filtered_cloud


def _split_floor_ceiling(
    points: np.ndarray,
    horizontal_mask: np.ndarray,
    tolerance: float,
):

    z = points[:, 2]

    horizontal_z = z[horizontal_mask]

    floor_level = np.min(horizontal_z)
    ceiling_level = np.max(horizontal_z)

    floor_mask = horizontal_mask & (
        z <= floor_level + tolerance
    )

    ceiling_mask = horizontal_mask & (
        z >= ceiling_level - tolerance
    )

    return floor_mask, ceiling_mask


def _crop_above_walls(
    cloud: o3d.geometry.PointCloud,
    walls: o3d.geometry.PointCloud,
    bin_size: float = 0.02,
) -> o3d.geometry.PointCloud:
    """
    Keep all points above the dominant wall height.

    The dominant wall height is estimated as the most common height
    among the highest wall points.
    """

    wall_points = np.asarray(walls.points)

    if len(wall_points) == 0:
        raise ValueError("No wall points detected.")

    wall_z = wall_points[:, 2]

    # We only analyse the upper part of the walls
    top_threshold = np.percentile(wall_z, 90)
    top_wall = wall_z[wall_z >= top_threshold]

    # Histogram
    bins = np.arange(
        top_wall.min(),
        top_wall.max() + bin_size,
        bin_size,
    )

    hist, edges = np.histogram(top_wall, bins=bins)

    peak = np.argmax(hist)

    wall_top = (edges[peak] + edges[peak + 1]) / 2

    points = np.asarray(cloud.points)

    mask = points[:, 2] >= wall_top

    return cloud.select_by_index(np.where(mask)[0])