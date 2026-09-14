from pathlib import Path
import sys

from src.pointcloud_io import (
    find_input_file,
    find_control_points_file,
    load_control_points,
    load_point_cloud,
    save_point_cloud,
)

from src.segmentation import segment_point_cloud
from src.transform import preprocess_point_cloud


def process_file(input_file: Path, project_dir: Path, control_points):
    """Process one point cloud file and save results in its own folder."""

    project_name = input_file.stem
    output_dir = project_dir / "output" / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing: {input_file.name} ===")
    print(f"Output directory: {output_dir}")

    print("Loading point cloud...")
    cloud = load_point_cloud(input_file)

    print("Transforming point cloud...")
    cloud = preprocess_point_cloud(cloud, control_points)

    print("Segmenting point cloud...")
    result = segment_point_cloud(cloud, control_points)

    print("Saving results...")

    save_point_cloud(
        result.floor,
        output_dir / f"{project_name}_floor{input_file.suffix}",
        input_file,
    )

    save_point_cloud(
        result.walls,
        output_dir / f"{project_name}_walls{input_file.suffix}",
        input_file,
    )

    save_point_cloud(
        result.ceiling,
        output_dir / f"{project_name}_ceiling{input_file.suffix}",
        input_file,
    )

    save_point_cloud(
        result.cropped,
        output_dir / f"{project_name}_cropped{input_file.suffix}",
        input_file,
    )

    print("Done")


def process_project(project_dir: Path):
    """Process the standard single input file for a project."""

    input_file = find_input_file(project_dir)

    print(f"Loading control points for {project_dir.name}...")
    control_points_file = find_control_points_file(project_dir)
    control_points = load_control_points(control_points_file)

    process_file(input_file, project_dir, control_points)


def process_all_las(project_dir: Path):
    """Process every LAS file directly inside the project directory."""

    las_files = sorted(project_dir.glob("*.las"))

    if not las_files:
        raise FileNotFoundError(f"No LAS files found in: {project_dir}")

    print(f"Found {len(las_files)} LAS file(s) in {project_dir}")

    print("Loading control points...")
    control_points_file = find_control_points_file(project_dir)
    control_points = load_control_points(control_points_file)

    for input_file in las_files:
        process_file(input_file, project_dir, control_points)

    print(f"\nProcessed {len(las_files)} LAS file(s).")


def main():

    if len(sys.argv) == 2:
        project_name = sys.argv[1]
        project_dir = Path("data") / project_name
        process_project(project_dir)
        return

    if len(sys.argv) == 3 and sys.argv[1] == "--all-las":
        project_dir = Path(sys.argv[2])
        process_all_las(project_dir)
        return

    print("Usage:")
    print("  python main.py <project_name>")
    print("  python main.py --all-las <project_dir>")


if __name__ == "__main__":
    main()
