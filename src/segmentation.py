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
    floor_height_tolerance: float = 0.50,
    ceiling_min_height: float = 2.00,
    wall_sor_neighbors: int = 30,
    wall_sor_std_ratio: float = 1.5,
) -> SegmentationResult:
    """Segment horizontal floor/ceiling surfaces and walls.

    The first control point provides the floor height reference. All large
    horizontal planes close to that height are classified as floor. A large
    horizontal plane is classified as ceiling only when it is at least
    ``ceiling_min_height`` above the floor reference. Horizontal planes in
    between remain unclassified by this stage.
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

    floor_reference_z = _get_floor_reference_z(
        points,
        large_horizontal_planes,
        control_points[0],
    )

    floor_indices, ceiling_indices = _classify_horizontal_planes(
        points,
        large_horizontal_planes,
        floor_reference_z=floor_reference_z,
        floor_height_tolerance=floor_height_tolerance,
        ceiling_min_height=ceiling_min_height,
    )

    floor_cloud = cloud.select_by_index(floor_indices)

    wall_cloud = cloud.select_by_index(np.where(wall_mask)[0])
    wall_cloud = _remove_wall_outliers(
        wall_cloud,
        nb_neighbors=wall_sor_neighbors,
        std_ratio=wall_sor_std_ratio,
    )

    ceiling_cloud = cloud.select_by_index(ceiling_indices)
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


def _get_floor_reference_z(
    points: np.ndarray,
    planes: list[np.ndarray],
    origin: np.ndarray,
) -> float:
    """Get the height of the horizontal plane closest to the origin point."""

    if not planes:
        raise ValueError("No large horizontal planes detected.")

    best_plane = None
    best_distance = np.inf

    for plane_indices in planes:
        plane_points = points[plane_indices]
        distances_xy = np.linalg.norm(
            plane_points[:, :2] - origin[:2],
            axis=1,
        )
        distance_xy = float(np.min(distances_xy))

        if distance_xy < best_distance:
            best_distance = distance_xy
            best_plane = plane_indices

    if best_plane is None:
        raise ValueError("Could not identify floor reference plane.")

    return float(np.median(points[best_plane, 2]))


def _classify_horizontal_planes(
    points: np.ndarray,
    planes: list[np.ndarray],
    floor_reference_z: float,
    floor_height_tolerance: float,
    ceiling_min_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify every detected horizontal plane as floor or ceiling.

    Multiple planes can belong to the floor when their height is close to the
    floor reference. A ceiling must be at least ``ceiling_min_height`` above
    the floor reference. Intermediate horizontal planes are ignored here.
    """

    floor_parts: list[np.ndarray] = []
    ceiling_parts: list[np.ndarray] = []

    ceiling_height = floor_reference_z + ceiling_min_height

    for plane_indices in planes:
        plane_z = float(np.median(points[plane_indices, 2]))

        if abs(plane_z - floor_reference_z) <= floor_height_tolerance:
            floor_parts.append(plane_indices)
        elif plane_z >= ceiling_height:
            ceiling_parts.append(plane_indices)

    floor_indices = (
        np.concatenate(floor_parts)
        if floor_parts
        else np.array([], dtype=int)
    )

    ceiling_indices = (
        np.concatenate(ceiling_parts)
        if ceiling_parts
        else np.array([], dtype=int)
    )

    return floor_indices, ceiling_indices


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
