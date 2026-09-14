#!/usr/bin/env python3
"""Interactive top-down elevator selector for corridor LAS point clouds.

Left-drag a rectangle around the elevator. Press Enter to confirm.
The selected rectangle is saved as JSON and contains the XY bounds and
center. No point-cloud data is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector


class ElevatorSelector:
    def __init__(self, ax: plt.Axes, points_xy: np.ndarray) -> None:
        self.ax = ax
        self.points_xy = points_xy
        self.rect: tuple[float, float, float, float] | None = None
        self.selector = RectangleSelector(
            ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=0.05,
            minspany=0.05,
            spancoords="data",
            interactive=True,
        )

    def _on_select(self, press, release) -> None:
        x0, x1 = sorted((float(press.xdata), float(release.xdata)))
        y0, y1 = sorted((float(press.ydata), float(release.ydata)))
        self.rect = (x0, y0, x1, y1)
        print(f"Elevator: X [{x0:.3f}, {x1:.3f}], Y [{y0:.3f}, {y1:.3f}]")

    def confirm(self, _event=None) -> None:
        if self.rect is None:
            print("Najpierw zaznacz prostokat windy.")
            return
        self.ax.set_title("Potwierdzono. Zamknij okno.")
        plt.close(self.ax.figure)


def load_xy(path: Path, max_points: int = 250_000) -> np.ndarray:
    las = laspy.read(path)
    xy = np.column_stack((las.x, las.y)).astype(np.float64)
    if len(xy) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(xy), size=max_points, replace=False)
        xy = xy[indices]
    return xy


def main() -> None:
    parser = argparse.ArgumentParser(description="Select elevator in a LAS point cloud")
    parser.add_argument("las", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/elevator_selection.json"),
    )
    args = parser.parse_args()

    if args.las.suffix.lower() != ".las":
        raise ValueError("Input musi byc w formacie LAS (.las).")
    if not args.las.exists():
        raise FileNotFoundError(args.las)

    xy = load_xy(args.las)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(xy[:, 0], xy[:, 1], s=0.25, alpha=0.45, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Zaznacz prostokat obejmujacy WINE. ENTER = potwierdz.")
    ax.grid(True, alpha=0.2)

    selector = ElevatorSelector(ax, xy)
    fig.canvas.mpl_connect("key_press_event", selector.confirm if False else lambda e: selector.confirm(e) if e.key == "enter" else None)
    plt.show()

    if selector.rect is None:
        raise SystemExit("Nie wybrano windy.")

    x0, y0, x1, y1 = selector.rect
    result = {
        "input_las": str(args.las.resolve()),
        "selection": {
            "type": "elevator",
            "x_min": x0,
            "y_min": y0,
            "x_max": x1,
            "y_max": y1,
            "center_x": (x0 + x1) / 2.0,
            "center_y": (y0 + y1) / 2.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Zapisano: {args.output}")


if __name__ == "__main__":
    main()
