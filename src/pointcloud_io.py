from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

SUPPORTED_EXTENSIONS = (".las",)


def find_input_file(project_dir: Path) -> Path:
    """Find the single LAS point cloud stored directly in the input folder."""
    las_files = sorted(project_dir.glob("*.las"))

    if not las_files:
        raise FileNotFoundError(f"No LAS point cloud found in: {project_dir}")

    if len(las_files) > 1:
        files = "\n".join(str(path) for path in las_files)
        raise ValueError(
            "Expected exactly one LAS point cloud in the input folder. Found:\n" + files
        )

    return las_files[0]


def find_control_points_file(project_dir: Path) -> Path:
    """Find control_points.txt directly in the input folder."""
    file_path = project_dir / "control_points.txt"

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

        # Preserve LAS RGB colors when present.
        if all(hasattr(las, channel) for channel in ("red", "green", "blue")):
            colors = np.column_stack((las.red, las.green, las.blue)).astype(np.float64)
            max_color = max(float(colors.max()), 1.0)
            colors /= 65535.0 if max_color > 255.0 else 255.0
            cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

        return cloud

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

        # Preserve RGB in generated LAS files when the input LAS contains it.
        if (
            all(hasattr(template, channel) for channel in ("red", "green", "blue"))
            and cloud.has_colors()
        ):
            colors = np.asarray(cloud.colors)
            colors = np.clip(colors, 0.0, 1.0)
            rgb = np.rint(colors * 65535.0).astype(np.uint16)
            las.red = rgb[:, 0]
            las.green = rgb[:, 1]
            las.blue = rgb[:, 2]

        las.write(output_path)
        return

    raise ValueError(f"Unsupported output format: {suffix}")
