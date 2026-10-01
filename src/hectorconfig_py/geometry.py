"""Probe polygons, overlap diagnostics, and field-of-view checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import HectorConstants


def cosd(value):
    """Cosine with degree input, corresponding to the R engine's ``cosd``."""
    return np.cos(np.deg2rad(value))


def sind(value):
    """Sine with degree input, corresponding to the R engine's ``sind``."""
    return np.sin(np.deg2rad(value))


def asind(value):
    """Inverse sine returning degrees."""
    return np.rad2deg(np.arcsin(value))


def atand(value):
    """Inverse tangent returning degrees."""
    return np.rad2deg(np.arctan(value))


def atan2d(y, x):
    """Two-argument inverse tangent returning degrees."""
    return np.rad2deg(np.arctan2(y, x))


def rotate_points(points: np.ndarray, theta: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    """Rotate an ``(N, 2)`` array counter-clockwise around ``(x, y)``.

    The R code rotates each polygon about its probe head.  We construct the
    polygon at the origin, rotate it once, and then translate it to the
    requested head position.
    """

    points = np.asarray(points, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    shifted = points - np.array([x, y])
    return shifted @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def _head_angles(tip_l: float, tip_w: float, delta_poly: float) -> np.ndarray:
    """Angles used to approximate the rounded probe head with straight edges."""
    start = -atand(tip_l / tip_w) + delta_poly
    stop = 180.0 + atand(tip_l / tip_w) - delta_poly
    return np.arange(start, stop + delta_poly * 0.1, delta_poly)


def _probe_local_polygon(constants: HectorConstants, guide: bool = False) -> np.ndarray:
    """Build one unrotated exclusion polygon centred on the probe head.

    The first part is the rounded head; the remaining points describe the
    rectangular ferrule and cable.  Keeping this in local coordinates makes
    the science and guide shapes differ only through their dimensions.
    """
    if guide:
        tip_w, tip_l = constants.gtip_w, constants.gtip_l
        probe_w, probe_l = constants.gprobe_w, constants.gprobe_l
        cable_w, cable_l = constants.gcable_w, constants.gcable_l
        exclusion = constants.gexcl_radius
    else:
        tip_w, tip_l = constants.tip_w, constants.tip_l
        probe_w, probe_l = constants.probe_w, constants.probe_l
        cable_w, cable_l = constants.cable_w, constants.cable_l
        exclusion = constants.excl_radius

    rbuff = 2.0 - cosd(constants.delta_poly / 2.0)
    angles = _head_angles(tip_l, tip_w, constants.delta_poly)
    head_x = rbuff * exclusion * sind(angles)
    head_y = -rbuff * exclusion * cosd(angles)
    # The legacy engine uses slightly different x-anchors for the two probe
    # families.  Science probes start at ``-tip_l / 2``; guide probes start at
    # ``-(gtip_l - gtip_w / 2)``.  They happen to be numerically similar for
    # the current guide dimensions, but they are not the same expression and
    # must remain distinct when constants change.
    base_x = -tip_l / 2.0 if not guide else -(tip_l - tip_w / 2.0)
    body_x = [
        base_x, base_x, base_x - probe_l, base_x - probe_l,
        base_x - probe_l - cable_l, base_x - probe_l - cable_l,
        base_x - probe_l, base_x - probe_l, base_x, base_x,
    ]
    body_y = [
        tip_w / 2.0, probe_w / 2.0, probe_w / 2.0, cable_w / 2.0,
        cable_w / 2.0, -cable_w / 2.0, -cable_w / 2.0,
        -probe_w / 2.0, -probe_w / 2.0, -tip_w / 2.0,
    ]
    return np.column_stack([
        np.concatenate([head_x, body_x]),
        np.concatenate([head_y, body_y]),
    ])


def probe_polygon(x: float, y: float, angle: float, constants: HectorConstants, guide: bool = False) -> np.ndarray:
    """Return one complete exclusion polygon in millimetres.

    ``angle`` is in radians, matching the R engine's internal representation.
    The returned polygon is the *exclusion* shape used for collision tests,
    not merely the visible ferrule outline.
    """

    return rotate_points(_probe_local_polygon(constants, guide), angle, 0.0, 0.0) + np.array([x, y])


def probe_footprint(x: float, y: float, angle: float, constants: HectorConstants, guide: bool = False) -> np.ndarray:
    """Return the narrower physical footprint used for plotting.

    The R engine used a larger exclusion polygon for safety and a smaller
    footprint for display.  Retaining both prevents the diagnostic plot from
    implying that a mathematically safe clearance is larger than it is.
    """

    polygon = _probe_local_polygon(constants, guide)
    tip_w = constants.gtip_w if guide else constants.tip_w
    footprint = np.vstack([
        [tip_w / 2.0, tip_w / 2.0],
        polygon[-10:],
        [tip_w / 2.0, -tip_w / 2.0],
    ])
    return rotate_points(footprint, angle) + np.array([x, y])


def make_polygons(positions: pd.DataFrame, angles: Iterable[float], constants: HectorConstants, guide: bool = False) -> list[np.ndarray]:
    """Create one exclusion polygon for every row in a position table."""
    angles = list(angles)
    if len(positions) != len(angles):
        raise ValueError("There must be one angle for every probe position.")
    return [
        probe_polygon(float(row.x), float(row.y), float(angle), constants, guide)
        for row, angle in zip(positions.itertuples(), angles)
    ]


def _shape(points: np.ndarray):
    try:
        from shapely.geometry import Polygon
    except ImportError as exc:  # pragma: no cover - exercised when optional dep absent
        raise ImportError(
            "Geometry conflict checks need Shapely. Install the package with "
            "`python -m pip install -e '.[app]'`."
        ) from exc
    shape = Polygon(points)
    return shape if shape.is_valid else shape.buffer(0)


def intersection_area(first: np.ndarray, second: np.ndarray) -> float:
    """Return the overlap area of two polygons in square millimetres."""
    return float(_intersection(first, second).area)


def _intersection(first: np.ndarray, second: np.ndarray):
    """Return the Shapely intersection used by the R ``joinPolys`` analogue."""
    return _shape(first).intersection(_shape(second))


def point_to_polygon_distances(points: pd.DataFrame, polygons: list[np.ndarray]) -> np.ndarray:
    """Return each point's distance from the nearest polygon.

    This is the operation used by the R engine's standard and guide cleaning
    helpers. A point inside a polygon has distance zero and is therefore
    rejected by the same strict exclusion-radius test as in R.
    """
    try:
        from shapely.geometry import Point
    except ImportError as exc:  # pragma: no cover - exercised when optional dep absent
        raise ImportError(
            "Geometry conflict checks need Shapely. Install the package with "
            "`python -m pip install -e '.[app]'`."
        ) from exc
    shapes = [_shape(polygon) for polygon in polygons]
    return np.asarray(
        [
            min(
                (shape.distance(Point(float(row.x), float(row.y))) for shape in shapes),
                default=np.inf,
            )
            for row in points.itertuples()
        ],
        dtype=float,
    )


def find_probe_conflicts(polygons: list[np.ndarray]) -> pd.DataFrame:
    """Return pairwise target/standard conflicts using one-based R-style IDs.

    Conflict rows deliberately use one-based probe numbers because those are
    the numbers in the original R diagnostics and are easier to compare while
    validating the two implementations.
    """

    rows: list[dict[str, float]] = []
    areas = [max(_shape(poly).area, np.finfo(float).eps) for poly in polygons]
    for i in range(len(polygons) - 1):
        for j in range(i + 1, len(polygons)):
            intersection = _intersection(polygons[i], polygons[j])
            area = float(intersection.area)
            # R treats any non-null ``joinPolys(..., operation='INT')`` as a
            # conflict, including exact edge/point contact. Keep that strict
            # behaviour rather than silently allowing touching probes.
            if not intersection.is_empty:
                rows.append({
                    "Probe_1st": i + 1,
                    "Probe_2nd": j + 1,
                    "PercentArea": round(100.0 * area / areas[i], 1),
                })
    return pd.DataFrame(rows, columns=["Probe_1st", "Probe_2nd", "PercentArea"])


def find_guide_conflicts(target_polygons: list[np.ndarray], guide_polygons: list[np.ndarray]) -> pd.DataFrame:
    """Find target-guide and guide-guide overlaps in one table."""
    rows: list[dict[str, object]] = []
    target_areas = [max(_shape(poly).area, np.finfo(float).eps) for poly in target_polygons]
    for i, target in enumerate(target_polygons):
        for j, guide in enumerate(guide_polygons):
            intersection = _intersection(target, guide)
            area = float(intersection.area)
            if not intersection.is_empty:
                rows.append({
                    "Probe_1st": i + 1,
                    "Probe_2nd": j + 1,
                    "PercentArea": round(100.0 * area / target_areas[i], 1),
                    "ConflictType": "target-guide",
                })
    guide_areas = [max(_shape(poly).area, np.finfo(float).eps) for poly in guide_polygons]
    for i in range(len(guide_polygons) - 1):
        for j in range(i + 1, len(guide_polygons)):
            intersection = _intersection(guide_polygons[i], guide_polygons[j])
            area = float(intersection.area)
            if not intersection.is_empty:
                rows.append({
                    "Probe_1st": i + 1,
                    "Probe_2nd": j + 1,
                    "PercentArea": round(100.0 * area / guide_areas[i], 1),
                    "ConflictType": "guide-guide",
                })
    return pd.DataFrame(rows, columns=["Probe_1st", "Probe_2nd", "PercentArea", "ConflictType"])


def fov_status(polygons: list[np.ndarray], radius: float, tolerance: float = 1e-7) -> dict[str, object]:
    """Check every polygon vertex against the circular field boundary.

    A circle is convex, so checking all vertices is sufficient for these
    straight polygon edges.  This check is intentionally separate from the
    centre-position filter: a probe head can be inside the field while its
    cable polygon extends outside it.  A small positive clearance is required;
    a polygon that merely touches the boundary is rejected.
    """
    outside: list[int] = []
    clearances: list[float] = []
    for i, polygon in enumerate(polygons):
        if not np.all(np.isfinite(polygon)):
            outside.append(i + 1)
            clearances.append(float("-inf"))
            continue
        maximum_radius = float(np.sqrt(np.sum(polygon ** 2, axis=1).max()))
        clearance = radius - maximum_radius
        clearances.append(clearance)
        # ``<=`` is deliberate: exact contact with the plate is forbidden.
        if clearance <= tolerance:
            outside.append(i + 1)
    return {
        "ok": not outside,
        "outside": outside,
        "minimum_clearance_mm": min(clearances, default=float("inf")),
        "clearance_mm": clearances,
    }


def ferule_tip_positions(
    positions: pd.DataFrame,
    angles: Iterable[float],
    constants: HectorConstants,
    *,
    guide: bool = False,
) -> np.ndarray:
    """Return the outer ferule tips used by the R wedge check.

    The stored angle points along the probe's local positive axis.  The cable
    and ferule extend in the opposite direction, so the tip is displaced by
    ``probe_l + tip_l / 2`` along ``-angle``.  This physical plugability
    diagnostic is separate from the polygon-overlap test.
    """

    positions = positions[["x", "y"]].reset_index(drop=True).astype(float)
    angles = np.asarray(list(angles), dtype=float)
    if len(positions) != len(angles):
        raise ValueError("There must be one angle for every ferule position.")
    length = (
        constants.gprobe_l + constants.gtip_l / 2.0
        if guide
        else constants.probe_l + constants.tip_l / 2.0
    )
    direction = np.column_stack([np.cos(angles), np.sin(angles)])
    return positions[["x", "y"]].to_numpy() - length * direction


def _minor_arc_contains(angle: float, first: float, second: float) -> bool:
    """Whether ``angle`` lies strictly between two angles on the short arc."""

    two_pi = 2.0 * np.pi
    first = float(first) % two_pi
    second = float(second) % two_pi
    angle = float(angle) % two_pi
    delta = (second - first) % two_pi
    if delta > np.pi:
        first, second = second, first
        delta = (second - first) % two_pi
    distance = (angle - first) % two_pi
    return 0.0 < distance < delta


def wedge_status(
    target_positions: pd.DataFrame,
    target_angles: Iterable[float],
    guide_positions: pd.DataFrame | None,
    guide_angles: Iterable[float] | None,
    constants: HectorConstants,
) -> dict[str, object]:
    """Port the R ``check_wedge_issue`` physical pinch diagnostic.

    Two nearby probe heads can form a pinch through which a third probe's
    ferrule cannot safely pass.  The R code flags a third probe when its tip
    lies within the bend radius of both pinch tips and its current direction
    points into the narrow wedge.  Such a configuration may have no polygon
    overlap and still be physically difficult or impossible to plug.
    """

    target_positions = target_positions[["x", "y"]].reset_index(drop=True).astype(float)
    target_angles = np.asarray(list(target_angles), dtype=float)
    if guide_positions is None:
        guide_positions = pd.DataFrame(columns=["x", "y"], dtype=float)
    guide_positions = guide_positions[["x", "y"]].reset_index(drop=True).astype(float)
    guide_angles = np.asarray(
        [] if guide_angles is None else list(guide_angles), dtype=float
    )
    if len(target_positions) != len(target_angles):
        raise ValueError("There must be one target angle for every target position.")
    if len(guide_positions) != len(guide_angles):
        raise ValueError("There must be one guide angle for every guide position.")

    all_positions = pd.concat([target_positions, guide_positions], ignore_index=True)
    all_angles = np.concatenate([target_angles, guide_angles])
    target_count = len(target_positions)
    all_kinds = ["target"] * target_count + ["guide"] * len(guide_positions)
    if len(all_positions):
        all_tips = np.vstack([
            ferule_tip_positions(target_positions, target_angles, constants),
            ferule_tip_positions(guide_positions, guide_angles, constants, guide=True),
        ])
    else:
        all_tips = np.empty((0, 2))

    # This follows the R implementation: the pinch threshold is deliberately
    # based on the science-head exclusion radius and cable width for all
    # target/guide combinations.
    pinch_distance = 2.0 * constants.excl_radius + constants.cable_w
    pinch_pairs: list[tuple[int, int]] = []
    for first in range(len(all_positions) - 1):
        for second in range(first + 1, len(all_positions)):
            separation = float(
                np.hypot(
                    all_positions.iloc[first].x - all_positions.iloc[second].x,
                    all_positions.iloc[first].y - all_positions.iloc[second].y,
                )
            )
            if separation < pinch_distance:
                pinch_pairs.append((first, second))

    violations: list[dict[str, object]] = []
    for first, second in pinch_pairs:
        for third in range(len(all_positions)):
            if third in (first, second):
                continue
            near_first = np.hypot(*(all_tips[third] - all_tips[first])) < constants.fbr
            near_second = np.hypot(*(all_tips[third] - all_tips[second])) < constants.fbr
            if not (near_first and near_second):
                continue
            flipped = (all_angles[third] + np.pi) % (2.0 * np.pi)
            if not _minor_arc_contains(flipped, all_angles[first], all_angles[second]):
                continue
            violations.append({
                "pinch": (first + 1, second + 1),
                "pinched_kind": all_kinds[third],
                "pinched_index": (
                    third + 1 if third < target_count else third - target_count + 1
                ),
                "nodes": [
                    (all_kinds[first], first if first < target_count else first - target_count),
                    (all_kinds[second], second if second < target_count else second - target_count),
                    (all_kinds[third], third if third < target_count else third - target_count),
                ],
            })
    return {
        "ok": not violations,
        "pinches": pinch_pairs,
        "violations": violations,
    }
