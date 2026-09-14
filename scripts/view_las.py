#!/usr/bin/env python3
"""Visualize an LAS point cloud with RGB preserved and light voxel downsampling."""

from __future__ import annotations

import argparse

import laspy
import numpy as np
import open3d as o3d


def load_las(path: str, voxel_size: float) -> o3d.geometry.PointCloud:
    las = laspy.read(path)
    points = np.column_stack((las.x, las.y, las.z))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if all(hasattr(las, channel) for channel in ("red", "green", "blue")):
        rgb = np.column_stack((las.red, las.green, las.blue)).astype(np.float64)
        rgb /= 65535.0 if rgb.max() > 255 else 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))

    return pcd.voxel_down_sample(voxel_size=voxel_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("las_path", help="Path to the LAS file")
    parser.add_argument("--voxel", type=float, default=0.03)
    args = parser.parse_args()

    pcd = load_las(args.las_path, args.voxel)
    print(f"Punkty po downsample: {len(pcd.points)}")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="LAS - raw diagnostic",
        width=1600,
        height=900,
    )


if __name__ == "__main__":
    main()
