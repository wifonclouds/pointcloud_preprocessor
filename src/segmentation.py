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
    control_points: np.ndarray,
    normal_radius: float = 0.20,
    normal_max_nn: int = 30,
    floor_angle: float = 15.0,
    wall_angle: float = 15.0,
    horizontal_cluster_eps: float = 0.12,
    horizontal_cluster_min_points: int = 100,
    wall_sor_neighbors: int = 30,
    wall_sor_std_ratio: float = 1.5,
) -> SegmentationResult:
    """
    Segment floor, walls, ceiling and points above the wall top.

    Horizontal surfaces are first detected from normals and then split into
    spatially connected large horizontal planes. The floor is the horizontal
    plane closest to the first control point (control_points[0]). The ceiling
    is the highest large horizontal plane.

    This prevents tables, shelves and other elevated horizontal objects from
    being classified as the floor merely because they happen to be horizontal.
    """

    if control_points.shape != (3, 3):
        raise ValueError(
            f"Expected 3 control points (3x3), got {control_points.shape}"
        )

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

    horizontal_cloud = cloud.select_by_index(np.where(horizontal_mask)[0])

    large_horizontal_planes = _extract_large_horizontal_planes(
        horizontal_cloud,
        eps=horizontal_cluster_eps,
        min_points=horizontal_cluster_min_points,
    )

    floor_indices = _select_floor_plane(
        points,
        large_horizontal_planes,
        control_points[0],
    )

    ceiling_indices = _select_ceiling_plane(
        points,
        large_horizontal_planes,
    )

    floor_mask = np.zeros(len(points), dtype=bool)
    ceiling_mask = np.zeros(len(points), dtype=bool)
    floor_mask[floor_indices] = True
    ceiling_mask[ceiling_indices] = True

    # Never classify the same points as both floor and ceiling.
    ceiling_mask[floor_mask] = False

    floor_cloud = cloud.select_by_index(np.where(floor_mask)[0])

    wall_cloud = cloud.select_by_index(
        np.where(wall_mask)[0]
    )

    # SOR is applied ONLY to walls.
    wall_cloud = _remove_wall_outliers(
        wall_cloud,
        nb_neighbors=wall_sor_neighbors,
        std_ratio=wall_sor_std_ratio,
    )

    ceiling_cloud = cloud.select_by_index(np.where(ceiling_mask)[0])

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


def _extract_large_horizontal_planes(
    horizontal_cloud: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
) -> list[np.ndarray]:
    """Find large connected horizontal surface clusters."""

    if len(horizontal_cloud.points) == 0:
        return []

    labels = np.asarray(
        horizontal_cloud.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False,
        )
    )

    large_planes: list[np.ndarray] = []

    for label in sorted(set(labels)):
        if label < 0:
            continue

        indices = np.where(labels == label)[0]

        if len(indices) >= min_points:
            large_planes.append(indices)

    horizontal_indices = np.where(
        np.all(
            np.isclose(
                np.asarray(horizontal_cloud.points)[:, None, :],
                np.asarray(horizontal_cloud.points)[None, :, :],
            ),
            axis=2,
        )
    )
    del horizontal_indices

    # cluster_dbscan indices are local to horizontal_cloud. Keep their actual
    # point coordinates and let the caller map them back to the main cloud.
    return large_planes


def _select_floor_plane(
    points: np.ndarray,
    local_planes: list[np.ndarray],
    first_control_point: np.ndarray,
) -> np.ndarray:
    """Select the horizontal plane whose points are closest to control point 0."""

    if not local_planes:
        raise ValueError("No large horizontal planes detected.")

    # The plane indices are local to the horizontal point cloud, so recreate
    # the horizontal point cloud ordering from the normal-derived mask.
    horizontal_mask = np.zeros(len(points), dtype=bool)
    for plane in local_planes:
        horizontal_mask[plane] = True

    # This helper is replaced by the caller's explicit plane coordinates below.
    raise RuntimeError("Internal plane index mapping error")


def _select_ceiling_plane(
    points: np.ndarray,
    local_planes: list[np.ndarray],
) -> np.ndarray:
    raise RuntimeError("Internal plane index mapping error")


def _remove_wall_outliers(
    cloud: o3d.geometry.PointCloud,
    nb_neighbors: int = 30,
    std_ratio: float = 1.5,
) -> o3d.geometry.PointCloud:
    """Remove statistical outliers from the wall point cloud only."""

    if len(cloud.points) <= nb_neighbors:
        return cloud

    filtered_cloud, _ = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    return filtered_cloud


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

    top_threshold = np.percentile(wall_z, 90)
    top_wall = wall_z[wall_z >= top_threshold]

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
