#!/usr/bin/env python3
"""Top-down LAS diagnostic viewer for the corridor pipeline."""

from __future__ import annotations

import argparse

import laspy
import matplotlib.pyplot as plt
import numpy as np


def load_las(path: str, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    las = laspy.read(path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

    rgb = None
    if all(hasattr(las, channel) for channel in ("red", "green", "blue")):
        rgb = np.column_stack((las.red, las.green, las.blue)).astype(np.float64)
        rgb /= 65535.0 if rgb.max() > 255 else 255.0
        rgb = np.clip(rgb, 0.0, 1.0)

    if len(xyz) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]
        if rgb is not None:
            rgb = rgb[idx]

    return xyz, rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-down LAS diagnostic viewer")
    parser.add_argument("las_path")
    parser.add_argument("--max-points", type=int, default=300_000)
    parser.add_argument(
        "--all-heights",
        action="store_true",
        help="Show all heights instead of the automatic floor-level band",
    )
    parser.add_argument("--z-low", type=float, default=None)
    parser.add_argument("--z-high", type=float, default=None)
    args = parser.parse_args()

    xyz, rgb = load_las(args.las_path, args.max_points)

    if args.all_heights:
        mask = np.ones(len(xyz), dtype=bool)
        title = "LAS — top view — all heights"
    else:
        z_low = np.percentile(xyz[:, 2], 2.0) if args.z_low is None else args.z_low
        z_high = np.percentile(xyz[:, 2], 35.0) if args.z_high is None else args.z_high
        mask = (xyz[:, 2] >= z_low) & (xyz[:, 2] <= z_high)
        title = f"LAS — top view — floor band Z [{z_low:.2f}, {z_high:.2f}]"
        print(f"Automatyczny pas podłogi: Z [{z_low:.3f}, {z_high:.3f}]")

    points = xyz[mask]
    colors = rgb[mask] if rgb is not None else None

    print(f"Punkty wejściowe: {len(xyz)}")
    print(f"Punkty do wizualizacji: {len(points)}")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=0.35,
        c=colors,
        alpha=0.65,
        linewidths=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
