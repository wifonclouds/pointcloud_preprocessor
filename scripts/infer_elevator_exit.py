#!/usr/bin/env python3
"""Infer an elevator exit from floor continuity in a corridor LAS cloud.

Raw point density is intentionally not used: walls and fixtures can have high
point density. An open exit is expected to have continuous floor samples
extending away from one elevator side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np

SIDES = ("left", "right", "bottom", "top")
DIRECTIONS = {
    "left": np.array([-1.0, 0.0]),
    "right": np.array([1.0, 0.0]),
    "bottom": np.array([0.0, -1.0]),
    "top": np.array([0.0, 1.0]),
}


def floor_xy(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate the dominant floor height and return nearby XY samples."""
    z = xyz[:, 2]
    lo, hi = np.percentile(z, [1.0, 35.0])
    bins = np.arange(lo, hi + 0.02, 0.02)
    hist, edges = np.histogram(z, bins=bins)
    i = int(np.argmax(hist))
    z0 = float((edges[i] + edges[i + 1]) / 2.0)
    mask = np.abs(z - z0) <= 0.05
    return xyz[mask, :2], z0


def score_side(xy: np.ndarray, rect: tuple[float, float, float, float], side: str) -> float:
    """Measure continuous floor occupancy outside one elevator side."""
    x0, y0, x1, y1 = rect
    width, height = max(x1 - x0, 0.1), max(y1 - y0, 0.1)
    along = height if side in ("left", "right") else width
    depth = float(np.clip(along * 2.0, 1.0, 3.0))
    cell = 0.10
    margin = 0.08

    if side in ("left", "right"):
        a = xy[:, 1]
        if side == "left":
            d = x0 - xy[:, 0]
        else:
            d = xy[:, 0] - x1
        a0, a1 = y0 - margin, y1 + margin
    else:
        a = xy[:, 0]
        if side == "bottom":
            d = y0 - xy[:, 1]
        else:
            d = xy[:, 1] - y1
        a0, a1 = x0 - margin, x1 + margin

    mask = (d >= margin) & (d <= depth) & (a >= a0) & (a <= a1)
    a, d = a[mask], d[mask]
    if len(a) == 0:
        return 0.0

    na = max(1, int(np.ceil((a1 - a0) / cell)))
    nd = max(1, int(np.ceil((depth - margin) / cell)))
    ai = np.floor((a - a0) / cell).astype(int)
    di = np.floor((d - margin) / cell).astype(int)
    valid = (ai >= 0) & (ai < na) & (di >= 0) & (di < nd)
    grid = np.zeros((nd, na), dtype=bool)
    grid[di[valid], ai[valid]] = True

    # Coverage and persistence both matter. A few points at the door are not
    # enough; the floor should continue through a meaningful depth.
    coverage = float(grid.mean())
    depth_fraction = float(np.mean(np.any(grid, axis=1)))
    weighted = 0.0
    weights = np.linspace(1.0, 0.35, nd)
    for row, w in zip(grid, weights):
        weighted += w * float(row.mean())
    weighted /= float(weights.sum())
    return 0.35 * coverage + 0.40 * depth_fraction + 0.25 * weighted


def infer(xyz: np.ndarray, rect: tuple[float, float, float, float]) -> tuple[str, dict[str, float], float]:
    xy, z0 = floor_xy(xyz)
    scores = {side: score_side(xy, rect, side) for side in SIDES}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score < 0.08 or best_score < second_score * 1.10:
        raise RuntimeError(
            "Wyjscie windy jest niejednoznaczne. "
            f"scores={scores}; floor_z={z0:.3f}"
        )
    return best, scores, z0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("las", type=Path)
    p.add_argument("selection", type=Path)
    p.add_argument("--output", type=Path, default=Path("data/output/elevator_exit.json"))
    args = p.parse_args()

    if args.las.suffix.lower() != ".las":
        raise ValueError("Wejscie musi byc LAS (.las).")

    s = json.loads(args.selection.read_text(encoding="utf-8"))["selection"]
    rect = (float(s["x_min"]), float(s["y_min"]), float(s["x_max"]), float(s["y_max"]))

    las = laspy.read(args.las)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    if len(xyz) > 500_000:
        rng = np.random.default_rng(42)
        xyz = xyz[rng.choice(len(xyz), 500_000, replace=False)]

    side, scores, z0 = infer(xyz, rect)
    print(f"Estimated floor Z: {z0:.3f}")
    for name in SIDES:
        print(f"{name:6s}: {scores[name]:.4f}")
    print(f"EXIT: {side}")

    x0, y0, x1, y1 = rect
    origin = {
        "left": [x0, (y0 + y1) / 2],
        "right": [x1, (y0 + y1) / 2],
        "bottom": [(x0 + x1) / 2, y0],
        "top": [(x0 + x1) / 2, y1],
    }[side]
    result = {
        "input_las": str(args.las.resolve()),
        "exit_side": side,
        "origin_xy": origin,
        "exit_direction_xy": DIRECTIONS[side].tolist(),
        "floor_z": z0,
        "scores": scores,
        "convention": {"exit_direction": "+Y", "left": "-X"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Zapisano: {args.output}")


if __name__ == "__main__":
    main()
