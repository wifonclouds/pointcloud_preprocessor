#!/usr/bin/env python3
"""Compute a fixed origin from the selected elevator rectangle.

No rotation is performed. The original LAS coordinate axes are preserved.
The origin is placed at 2/3 of the elevator width measured from x_min to
x_max, with y at the center of the selected elevator rectangle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Set corridor origin from elevator selection")
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/output/origin.json"))
    args = parser.parse_args()

    data = json.loads(args.selection.read_text(encoding="utf-8"))
    selection = data["selection"]

    x_min = float(selection["x_min"])
    x_max = float(selection["x_max"])
    y_min = float(selection["y_min"])
    y_max = float(selection["y_max"])

    width = x_max - x_min
    if width <= 0 or y_max <= y_min:
        raise ValueError("Nieprawidlowe zaznaczenie windy.")

    origin_x = x_min + (2.0 / 3.0) * width
    origin_y = (y_min + y_max) / 2.0

    result = {
        "source_selection": str(args.selection.resolve()),
        "origin": {
            "x": origin_x,
            "y": origin_y,
            "z": 0.0,
        },
        "definition": {
            "x_position": "2/3 of elevator width from x_min toward x_max",
            "y_position": "elevator center",
            "rotation": "none",
        },
        "coordinate_convention": "original LAS axes preserved",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Elevator width: {width:.3f} m")
    print(f"Origin X: {origin_x:.3f}")
    print(f"Origin Y: {origin_y:.3f}")
    print("Rotation: NONE")
    print(f"Zapisano: {args.output}")


if __name__ == "__main__":
    main()
