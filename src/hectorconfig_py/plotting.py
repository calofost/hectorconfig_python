"""Matplotlib plots used by the command line workflow and Shiny app."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon as PolygonPatch
import numpy as np
import pandas as pd

from .constants import HectorConstants
from .geometry import make_polygons, probe_footprint


def _draw_probe(
    ax,
    polygon: np.ndarray,
    footprint: np.ndarray,
    color: str,
    label: str | None = None,
    label_position: tuple[float, float] | None = None,
):
    """Draw the probe and put its label at the probe-head centre.

    Labels used to be placed at the first footprint vertex, which is near the
    head edge and often overlapped the ferrule outline.  A small opaque circular
    label centred on ``(x, y)`` remains readable over the geometry.
    """
    ax.add_patch(PolygonPatch(polygon, closed=True, facecolor="none", edgecolor=color, linewidth=0.7, alpha=0.35))
    ax.add_patch(PolygonPatch(footprint, closed=True, facecolor="none", edgecolor=color, linewidth=1.0))
    if label:
        if label_position is None:
            label_position = tuple(np.mean(footprint[:2], axis=0))
        ax.text(
            label_position[0],
            label_position[1],
            label,
            fontsize=6,
            ha="center",
            va="center",
            color="black",
            zorder=10,
            bbox={
                "boxstyle": "circle,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.9,
            },
        )


def make_field_figure(
    target_positions: pd.DataFrame,
    target_angles: Iterable[float],
    guide_positions: pd.DataFrame | None,
    guide_angles: Iterable[float] | None,
    constants: HectorConstants,
    *,
    selected_target: int | None = None,
    selected_guide: int | None = None,
    standard_indices: Iterable[int] | None = None,
    compass_position: tuple[float, float] | None = None,
    compass_angle: float | None = None,
    compass_radius: float | None = None,
    title: str = "HECTOR configuration",
):
    """Build a Matplotlib field plot used by both CLI output and the app."""
    fig, ax = plt.subplots(figsize=(8, 8))
    radius = constants.plate_radius
    ax.add_patch(Circle((0, 0), radius, fill=False, color="red", linewidth=1.2))
    # The red circle is the hard plate boundary; the dashed circle is the
    # extra clear-edge buffer shown by the R diagnostic plot.
    ax.add_patch(Circle((0, 0), radius + constants.skybuffer, fill=False, color="black", linewidth=0.6, linestyle="--"))
    if standard_indices is None:
        # The automatic workflow stores science targets first and standards
        # last.  Callers that have retained the original row order can pass
        # explicit indices instead, which is what the Shiny app does.
        standard_set = set(range(
            max(0, len(target_positions) - constants.nstdprobes),
            len(target_positions),
        ))
    else:
        standard_set = {int(index) for index in standard_indices}

    for index, (position, angle, polygon) in enumerate(zip(
        target_positions.itertuples(), target_angles,
        make_polygons(target_positions, target_angles, constants),
    )):
        if index == selected_target:
            color = "tab:orange"
        elif index in standard_set:
            color = "tab:purple"
        else:
            color = "tab:blue"
        footprint = probe_footprint(position.x, position.y, angle, constants)
        _draw_probe(ax, polygon, footprint, color, str(index + 1), (position.x, position.y))
    if guide_positions is not None and guide_angles is not None:
        for index, (position, angle, polygon) in enumerate(zip(
            guide_positions.itertuples(), guide_angles,
            make_polygons(guide_positions, guide_angles, constants, guide=True),
        )):
            color = "tab:red" if index == selected_guide else "tab:green"
            footprint = probe_footprint(position.x, position.y, angle, constants, guide=True)
            _draw_probe(ax, polygon, footprint, color, f"G{index + 1}", (position.x, position.y))
    if compass_position is not None:
        _draw_angle_compass(
            ax,
            compass_position[0],
            compass_position[1],
            compass_radius or (constants.probe_l + constants.cable_l + constants.tip_l / 2.0),
            compass_angle,
            constants,
        )
    # Cable exits are shown separately so a manual angle change can be
    # interpreted relative to the three physical gaps.
    exits = constants.ceg_positions
    ax.scatter(exits[:, 0], exits[:, 1], marker="x", color="black", zorder=5)
    for index, (x, y, degree) in enumerate(exits, start=1):
        ax.text(x, y, f"CEG {index}", fontsize=7, ha="center", va="bottom")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-radius - 20, radius + 20)
    ax.set_ylim(-radius - 20, radius + 20)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    ax.legend(
        handles=[
            Line2D([], [], color="tab:blue", linewidth=2, label="Science target"),
            Line2D([], [], color="tab:purple", linewidth=2, label="Standard"),
            Line2D([], [], color="tab:green", linewidth=2, label="Guide"),
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
    )
    fig.tight_layout()
    return fig, ax


def _draw_angle_compass(
    ax,
    x: float,
    y: float,
    radius: float,
    angle: float | None,
    constants: HectorConstants,
) -> None:
    """Draw the R-app-style degree compass around the selected probe.

    The R manual app labels 0 degrees along the probe's local negative-x
    direction, which is the direction of the cable/tail when the stored angle
    is zero.  A click in the desired tail direction is therefore converted to
    ``pi + atan2(click_y - y, click_x - x)`` by the app.  Showing that same
    convention here makes the visual control and the saved angle agree.
    """
    ax.add_patch(
        Circle(
            (x, y),
            radius,
            fill=False,
            edgecolor="crimson",
            linewidth=0.8,
            linestyle=":",
            alpha=0.9,
            zorder=12,
        )
    )
    ax.scatter([x], [y], marker="+", color="crimson", s=35, zorder=13)

    degrees = np.arange(0.0, 360.0, 30.0)
    label_radius = radius + constants.tip_l
    for degree in degrees:
        radians = np.deg2rad(degree)
        # Match the R app: 0 is to the left, and the labels proceed around
        # the compass in the same orientation as its angle convention.
        label_x = x - label_radius * np.cos(radians)
        label_y = y - label_radius * np.sin(radians)
        ax.text(
            label_x,
            label_y,
            f"{int(degree)}",
            fontsize=6,
            ha="center",
            va="center",
            color="crimson",
            zorder=13,
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    # Reference ray for 0 degrees (the local negative-x/tail direction).
    ax.plot([x, x - radius], [y, y], color="crimson", linewidth=0.7, alpha=0.8, zorder=12)
    if angle is not None and np.isfinite(angle):
        # The physical tail points opposite the local +x angle used in the
        # stored probe rotation.
        tail_angle = angle + np.pi
        ax.annotate(
            "",
            xy=(x + radius * np.cos(tail_angle), y + radius * np.sin(tail_angle)),
            xytext=(x, y),
            arrowprops={"arrowstyle": "-|>", "color": "darkorange", "linewidth": 1.5},
            zorder=14,
        )


def plot_configured_field(
    target_positions: pd.DataFrame,
    target_angles: Iterable[float],
    guide_positions: pd.DataFrame,
    guide_angles: Iterable[float],
    constants: HectorConstants,
    filename: str | Path,
    *,
    flags: Iterable[str] = (),
    standard_indices: Iterable[int] | None = None,
) -> None:
    """Save a non-interactive diagnostic plot to PDF, PNG, or another format."""
    fig, _ = make_field_figure(
        target_positions, target_angles, guide_positions, guide_angles, constants,
        standard_indices=standard_indices,
        title=f"HECTOR configuration ({', '.join(flags) or 'none'})",
    )
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)
    plt.close(fig)
