#!/usr/bin/env python3
"""Infer the elevator exit direction and orient a LAS cloud to +Y.

The elevator rectangle is read from select_elevator.py output. The four
sides are scored by point density in a thin strip immediately outside the
selection; the side with the lowest density is treated as the open exit.
The exit midpoint becomes the XY origin and the outward exit direction is
rotated onto +Y. RGB is preserved in the output LAS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np


def load_selection(path: Path) -> tuple[float, float, float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["selection"]
    return float(s["x_min"]), float(s["y_min"]), float(s["x_max"]), float(s["y_max"])


def side_density(points: np.ndarray, rect: tuple[float, float, float, float], side: str) -> float:
    x0, y0, x1, y1 = rect
    # Work with XY only. The strip is deliberately local to the elevator.
    width = max(x1 - x0, 0.1)
    height = max(y1 - y0, 0.1)
    strip = max(min(width, height) * 0.45, 0.15)
    margin = max(min(width, height) * 0.08, 0.05)

    if side == "left":
        mask = (points[:, 0] >= x0 - strip) & (points[:, 0] <= x0 - margin) & (points[:, 1] >= y0 - margin) & (points[:, 1] <= y1 + margin)
        area = (strip - margin) * (height + 2 * margin)
    elif side == "right":
        mask = (points[:, 0] >= x1 + margin) & (points[:, 0] <= x1 + strip) & (points[:, 1] >= y0 - margin) & (points[:, 1] <= y1 + margin)
        area = (strip - margin) * (height + 2 * margin)
    elif side == "bottom":
        mask = (points[:, 1] >= y0 - strip) & (points[:, 1] <= y0 - margin) & (points[:, 0] >= x0 - margin) & (points[:, 0] <= x1 + margin)
        area = (strip - margin) * (width + 2 * margin)
    else:
        mask = (points[:, 1] >= y1 + margin) & (points[:, 1] <= y1 + strip) & (points[:, 0] >= x0 - margin) & (points[:, 0] <= x1 + margin)
        area = (strip - margin) * (width + 2 * margin)

    return float(mask.sum() / max(area, 1e-9))


def infer_exit(points_xy: np.ndarray, rect: tuple[float, float, float, float]) -> tuple[str, dict[str, float]]:
    densities = {side: side_density(points_xy, rect, side) for side in ("left", "right", "bottom", "top")}
    return min(densities, key=densities.get), densities


def rotation_to_positive_y(direction_xy: np.ndarray) -> np.ndarray:
    angle = np.arctan2(direction_xy[1], direction_xy[0])
    target = np.pi / 2.0
    theta = target - angle
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_las(input_path: Path, output_path: Path, rect: tuple[float, float, float, float], exit_side: str) -> dict:
    las = laspy.read(input_path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

    x0, y0, x1, y1 = rect
    centers = {
        "left": np.array([x0, (y0 + y1) / 2.0]),
        "right": np.array([x1, (y0 + y1) / 2.0]),
        "bottom": np.array([(x0 + x1) / 2.0, y0]),
        "top": np.array([(x0 + x1) / 2.0, y1]),
    }
    directions = {
        "left": np.array([-1.0, 0.0]),
        "right": np.array([1.0, 0.0]),
        "bottom": np.array([0.0, -1.0]),
        "top": np.array([0.0, 1.0]),
    }

    origin_xy = centers[exit_side]
    exit_direction = directions[exit_side]
    R = rotation_to_positive_y(exit_direction)

    shifted_xy = xyz[:, :2] - origin_xy
    transformed_xy = shifted_xy @ R.T
    transformed = xyz.copy()
    transformed[:, :2] = transformed_xy

    header = las.header.copy()
    out = laspy.LasData(header)
    out.x = transformed[:, 0]
    out.y = transformed[:, 1]
    out.z = transformed[:, 2]

    for name in ("red", "green", "blue"):
        if name in las.point_format.dimension_names:
            setattr(out, name, getattr(las, name))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(output_path)

    return {
        "input_las": str(input_path.resolve()),
        "output_las": str(output_path.resolve()),
        "elevator_exit_side": exit_side,
        "origin_xy": origin_xy.tolist(),
        "exit_direction_xy": exit_direction.tolist(),
        "rotation_matrix_2d": R.tolist(),
        "convention": {"exit_direction": "+Y", "left_of_exit": "-X"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("las", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/output/02_transformed.las"))
    parser.add_argument("--metadata", type=Path, default=Path("data/output/transformation.json"))
    args = parser.parse_args()

    if args.las.suffix.lower() != ".las":
        raise ValueError("Input musi byc w formacie LAS (.las).")

    rect = load_selection(args.selection)
    las = laspy.read(args.las)
    xy = np.column_stack((las.x, las.y)).astype(np.float64)
    if len(xy) > 300_000:
        rng = np.random.default_rng(42)
        xy = xy[rng.choice(len(xy), 300_000, replace=False)]

    exit_side, densities = infer_exit(xy, rect)
    print("Gestosc punktow za bokami windy:")
    for side, value in densities.items():
        print(f"  {side:6s}: {value:.2f}")
    print(f"Wybrane wyjscie: {exit_side}")

    metadata = transform_las(args.las, args.output, rect, exit_side)
    metadata["side_density"] = densities
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Zapisano: {args.output}")
    print(f"Zapisano: {args.metadata}")


if __name__ == "__main__":
    main()
