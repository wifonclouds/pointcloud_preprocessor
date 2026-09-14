from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

SUPPORTED_EXTENSIONS = (".las", ".ply", ".pcd")


def find_input_file(project_dir: Path) -> Path:
    input_dir = project_dir / "input"
    project_name = project_dir.name

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for extension in SUPPORTED_EXTENSIONS:
        input_file = input_dir / f"{project_name}{extension}"

        if input_file.exists():
            return input_file

    expected_files = "\n".join(
        str(input_dir / f"{project_name}{ext}")
        for ext in SUPPORTED_EXTENSIONS
    )

    raise FileNotFoundError(
        f"No input point cloud found.\nExpected one of:\n{expected_files}"
    )


def find_control_points_file(project_dir: Path) -> Path:
    """Find control_point.txt directly in the project directory."""
    file_path = project_dir / "control_point.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Control points file not found: {file_path}"
        )

    return file_path


def load_point_cloud(file_path: Path) -> o3d.geometry.PointCloud:
    suffix = file_path.suffix.lower()

    if suffix == ".las":
        las = laspy.read(file_path)

        points = np.column_stack((las.x, las.y, las.z))

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        return cloud

    elif suffix in (".ply", ".pcd"):
        return o3d.io.read_point_cloud(str(file_path))

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_control_points(file_path: Path) -> np.ndarray:
    try:
        control_points = np.loadtxt(file_path, dtype=float, delimiter=",")
    except ValueError:
        control_points = np.loadtxt(file_path, dtype=float)

    if control_points.shape != (3, 3):
        raise ValueError(
            f"Expected exactly 3 control points (3x3), got {control_points.shape}"
        )

    return control_points


def save_point_cloud(
    cloud: o3d.geometry.PointCloud,
    output_path: Path,
    template_path: Path | None = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix in (".ply", ".pcd"):
        o3d.io.write_point_cloud(str(output_path), cloud)
        return

    if suffix == ".las":
        if template_path is None:
            raise ValueError("template_path is required for LAS export.")

        template = laspy.read(template_path)

        las = laspy.create(
            point_format=template.header.point_format,
            file_version=template.header.version,
        )

        las.header.scales = template.header.scales
        las.header.offsets = template.header.offsets

        points = np.asarray(cloud.points)

        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        las.write(output_path)

        return

    raise ValueError(f"Unsupported output format: {suffix}")
