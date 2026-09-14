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


def main():

    if len(sys.argv) != 2:
        print("Usage: python main.py <project_name>")
        return

    project_name = sys.argv[1]
    project_dir = Path("data") / project_name

    print("Loading point cloud...")
    input_file = find_input_file(project_dir)
    cloud = load_point_cloud(input_file)

    print("Loading control points...")
    control_points_file = find_control_points_file(project_dir)
    control_points = load_control_points(control_points_file)

    print("Transforming point cloud...")
    cloud = preprocess_point_cloud(cloud, control_points)

    print("Segmenting point cloud...")
    result = segment_point_cloud(cloud)

    output_dir = project_dir / "output"

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
        result.cropped,
        output_dir / f"{project_name}_cropped{input_file.suffix}",
        input_file,
    )

    print("\nDone!")
    print(f"Floor   : {output_dir / f'{project_name}_floor{input_file.suffix}'}")
    print(f"Walls   : {output_dir / f'{project_name}_walls{input_file.suffix}'}")
    print(f"Cropped : {output_dir / f'{project_name}_cropped{input_file.suffix}'}")


if __name__ == "__main__":
    main()