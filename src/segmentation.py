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
    """Segment large horizontal planes, walls and the space above the walls.

    All detected large horizontal planes are preserved as ceiling candidates.
    The first control point is used only to identify the floor plane.
    """

    if control_points.shape != (3, 3):
        raise ValueError(
            f"Expected 3 control points (3x3), got {control_points.shape}"
        )

    cloud = cloud.voxel_down_sample(0.02)
    _estimate_normals(cloud, radius=normal_radius, max_nn=normal_max_nn)

    points = np.asarray(cloud.points)
    horizontal_mask = _detect_horizontal_surfaces(cloud, floor_angle)
    wall_mask = _detect_walls(cloud, wall_angle)

    horizontal_indices = np.where(horizontal_mask)[0]
    horizontal_cloud = cloud.select_by_index(horizontal_indices)

    large_horizontal_planes = _extract_large_horizontal_planes(
        horizontal_cloud,
        horizontal_indices,
        eps=horizontal_cluster_eps,
        min_points=horizontal_cluster_min_points,
    )

    floor_indices = _select_floor_plane(
        points,
        large_horizontal_planes,
        control_points[0],
    )

    # Keep EVERY large horizontal plane except the selected floor.
    ceiling_indices = _select_all_ceiling_planes(
        large_horizontal_planes,
        floor_indices,
    )

    floor_mask = np.zeros(len(points), dtype=bool)
    ceiling_mask = np.zeros(len(points), dtype=bool)
    floor_mask[floor_indices] = True
    ceiling_mask[ceiling_indices] = True

    floor_cloud = cloud.select_by_index(np.where(floor_mask)[0])

    wall_cloud = cloud.select_by_index(np.where(wall_mask)[0])
    wall_cloud = _remove_wall_outliers(
        wall_cloud,
        nb_neighbors=wall_sor_neighbors,
        std_ratio=wall_sor_std_ratio,
    )

    ceiling_cloud = cloud.select_by_index(np.where(ceiling_mask)[0])
    cropped_cloud = _crop_above_walls(cloud, wall_cloud)

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
    horizontal_indices: np.ndarray,
    eps: float,
    min_points: int,
) -> list[np.ndarray]:
    """Return large horizontal plane clusters as indices into the main cloud."""

    if len(horizontal_cloud.points) == 0:
        return []

    labels = np.asarray(
        horizontal_cloud.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False,
        )
    )

    planes: list[np.ndarray] = []

    for label in sorted(set(labels)):
        if label < 0:
            continue

        local_indices = np.where(labels == label)[0]
        if len(local_indices) < min_points:
            continue

        planes.append(horizontal_indices[local_indices])

    return planes


def _select_floor_plane(
    points: np.ndarray,
    planes: list[np.ndarray],
    first_control_point: np.ndarray,
) -> np.ndarray:
    """Select the large horizontal plane closest to control point 0."""

    if not planes:
        raise ValueError("No large horizontal planes detected.")

    best_plane = None
    best_distance = np.inf

    for plane_indices in planes:
        plane_points = points[plane_indices]
        distances = np.linalg.norm(plane_points - first_control_point, axis=1)
        distance = float(np.min(distances))

        if distance < best_distance:
            best_distance = distance
            best_plane = plane_indices

    if best_plane is None:
        raise ValueError("Could not identify floor plane from control point 0.")

    return best_plane


def _select_all_ceiling_planes(
    planes: list[np.ndarray],
    floor_indices: np.ndarray,
) -> np.ndarray:
    """Return all large horizontal planes except the selected floor plane."""

    floor_set = set(floor_indices.tolist())
    ceiling_parts = [
        plane for plane in planes
        if not set(plane.tolist()).issubset(floor_set)
    ]

    if not ceiling_parts:
        return np.array([], dtype=int)

    return np.concatenate(ceiling_parts)


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
    """Keep all points above the dominant wall height."""

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
