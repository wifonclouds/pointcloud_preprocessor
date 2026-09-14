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
    """Process one LAS point cloud and save results next to the input folder."""

    project_name = input_file.stem
    output_dir = project_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing: {input_file.name} ===")
    print(f"Input directory: {project_dir}")
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

    print("Done")


def process_project(project_dir: Path):
    """Process a folder containing exactly one LAS and control_points.txt."""

    if not project_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {project_dir}")

    input_file = find_input_file(project_dir)
    control_points_file = find_control_points_file(project_dir)

    print(f"Loading control points: {control_points_file}")
    control_points = load_control_points(control_points_file)

    process_file(input_file, project_dir, control_points)


def main():
    if len(sys.argv) == 2:
        process_project(Path(sys.argv[1]))
        return

    print("Usage:")
    print("  python main.py <input_folder>")
    print()
    print("Example:")
    print("  python main.py /home/kamil/Projects/pointcloud-preprocessor/data/dso7/output/dso7_apartment_f000/")


if __name__ == "__main__":
    main()
