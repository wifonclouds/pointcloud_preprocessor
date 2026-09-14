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
    wall_plane_distance: float = 0.05,
    wall_plane_min_points: int = 500,
    wall_min_height: float = 1.5,
    wall_min_horizontal_extent: float = 0.7,
    wall_min_vertical_coverage: float = 0.70,
    max_wall_planes: int = 100,
    wall_sor_neighbors: int = 30,
    wall_sor_std_ratio: float = 1.5,
) -> SegmentationResult:
    """Segment floor, structural walls and ceiling planes.

    Floor is the large horizontal plane closest to the first control point.
    Every other large horizontal plane above the floor is classified as
    ceiling.

    A wall is treated as structural only when a vertical RANSAC plane has
    enough points, enough horizontal extent and, importantly, extends through
    most of the floor-to-ceiling height. This prevents small vertical objects
    in the corridor from becoming walls.
    """

    if control_points.shape != (3, 3):
        raise ValueError(
            f"Expected 3 control points (3x3), got {control_points.shape}"
        )

    cloud = cloud.voxel_down_sample(0.02)
    _estimate_normals(cloud, radius=normal_radius, max_nn=normal_max_nn)

    points = np.asarray(cloud.points)
    horizontal_mask = _detect_horizontal_surfaces(cloud, floor_angle)

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
    )

    floor_cloud = cloud.select_by_index(floor_indices)

    ceiling_height = _get_ceiling_reference_z(points, ceiling_indices, floor_reference_z)

    wall_cloud = _extract_structural_walls(
        cloud,
        wall_angle=wall_angle,
        plane_distance=wall_plane_distance,
        min_points=wall_plane_min_points,
        min_height=wall_min_height,
        min_horizontal_extent=wall_min_horizontal_extent,
        floor_z=floor_reference_z,
        ceiling_z=ceiling_height,
        min_vertical_coverage=wall_min_vertical_coverage,
        max_planes=max_wall_planes,
    )

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


def _extract_structural_walls(
    cloud: o3d.geometry.PointCloud,
    wall_angle: float,
    plane_distance: float,
    min_points: int,
    min_height: float,
    min_horizontal_extent: float,
    floor_z: float,
    ceiling_z: float,
    min_vertical_coverage: float,
    max_planes: int,
) -> o3d.geometry.PointCloud:
    """Extract structural vertical planes using RANSAC and height coverage."""

    wall_mask = _detect_walls(cloud, wall_angle)
    wall_indices = np.where(wall_mask)[0]

    if len(wall_indices) < min_points:
        raise ValueError(
            f"Not enough vertical wall candidates: {len(wall_indices)} points."
        )

    remaining_indices = wall_indices.copy()
    points = np.asarray(cloud.points)
    accepted_indices: list[np.ndarray] = []

    if ceiling_z <= floor_z:
        ceiling_z = float(points[:, 2].max())

    target_height = ceiling_z - floor_z
    if target_height <= 0.0:
        raise ValueError("Invalid floor/ceiling height range.")

    # Divide the room height into bands. A real wall should be observed in
    # most bands, while a cabinet, sign, machine or other small object should
    # normally occupy only a subset of them.
    n_height_bins = 10
    lower = floor_z + 0.05 * target_height
    upper = ceiling_z - 0.05 * target_height
    bin_edges = np.linspace(lower, upper, n_height_bins + 1)
    required_bins = int(np.ceil(n_height_bins * min_vertical_coverage))

    while len(remaining_indices) >= min_points and len(accepted_indices) < max_planes:
        candidate = cloud.select_by_index(remaining_indices)

        plane_model, inliers = candidate.segment_plane(
            distance_threshold=plane_distance,
            ransac_n=3,
            num_iterations=1500,
        )

        if len(inliers) < min_points:
            break

        plane_normal = np.asarray(plane_model[:3], dtype=float)
        normal_length = np.linalg.norm(plane_normal)
        if normal_length == 0.0:
            break
        plane_normal /= normal_length

        # The fitted plane itself must be vertical.
        if abs(float(plane_normal[2])) > np.sin(np.deg2rad(wall_angle)):
            remaining_indices = np.delete(
                remaining_indices,
                np.asarray(inliers, dtype=int),
            )
            continue

        inlier_indices = remaining_indices[np.asarray(inliers, dtype=int)]
        plane_points = points[inlier_indices]

        extent = plane_points.max(axis=0) - plane_points.min(axis=0)
        horizontal_extent = max(float(extent[0]), float(extent[1]))
        height = float(extent[2])

        histogram, _ = np.histogram(plane_points[:, 2], bins=bin_edges)
        covered_bins = int(np.count_nonzero(histogram))

        # Structural wall criteria:
        # 1. enough points,
        # 2. physically tall,
        # 3. enough horizontal extent,
        # 4. observed through most of the floor-to-ceiling height.
        if (
            height >= min_height
            and horizontal_extent >= min_horizontal_extent
            and covered_bins >= required_bins
        ):
            accepted_indices.append(inlier_indices)

        remaining_indices = np.delete(
            remaining_indices,
            np.asarray(inliers, dtype=int),
        )

    if not accepted_indices:
        raise ValueError("No structural wall planes detected.")

    indices = np.unique(np.concatenate(accepted_indices))
    return cloud.select_by_index(indices)


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
    """Return the Z height of the horizontal plane nearest the origin in XY."""

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
) -> tuple[np.ndarray, np.ndarray]:
    """Classify the nearest horizontal plane as floor and all higher planes as ceiling."""

    floor_parts: list[np.ndarray] = []
    ceiling_parts: list[np.ndarray] = []
    floor_plane_found = False

    for plane_indices in planes:
        plane_z = float(np.median(points[plane_indices, 2]))

        if not floor_plane_found and np.isclose(
            plane_z,
            floor_reference_z,
            atol=0.01,
        ):
            floor_parts.append(plane_indices)
            floor_plane_found = True
        elif plane_z > floor_reference_z:
            ceiling_parts.append(plane_indices)

    if not floor_parts and planes:
        floor_plane = min(
            planes,
            key=lambda indices: abs(
                float(np.median(points[indices, 2])) - floor_reference_z
            ),
        )
        floor_parts.append(floor_plane)

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


def _get_ceiling_reference_z(
    points: np.ndarray,
    ceiling_indices: np.ndarray,
    floor_reference_z: float,
) -> float:
    """Return the highest detected ceiling plane, with a point-cloud fallback."""

    if len(ceiling_indices) == 0:
        return float(points[:, 2].max())

    return float(np.max(points[ceiling_indices, 2]))


def _remove_wall_outliers(
    cloud: o3d.geometry.PointCloud,
    nb_neighbors: int = 30,
    std_ratio: float = 1.5,
) -> o3d.geometry.PointCloud:
    """Remove statistical outliers from the structural wall cloud only."""

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
