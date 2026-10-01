"""Geometry and stochastic search translated from the stable R engine.

The search is deliberately explicit rather than hidden inside a mutable R-like
environment.  Each stage receives positions, angles, cable-exit choices, and
constants, then returns new arrays plus diagnostics.  This makes it possible
to compare Python and R after every stage rather than only comparing PDFs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import HectorConstants
from .geometry import (
    cosd,
    find_guide_conflicts,
    find_probe_conflicts,
    fov_status,
    intersection_area,
    make_polygons,
    sind,
    wedge_status,
)


def _positions(values: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Normalise an array or data frame to the two-column position format."""
    if isinstance(values, pd.DataFrame):
        try:
            return values.loc[:, ["x", "y"]].reset_index(drop=True).astype(float)
        except KeyError as exc:
            raise ValueError("Positions must contain columns named 'x' and 'y'.") from exc
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Positions must be a two-column x/y table or an (N, 2) array.")
    return pd.DataFrame(array, columns=["x", "y"])


def choose_cegs(positions: pd.DataFrame, constants: HectorConstants) -> np.ndarray:
    """Choose the least crowded cable exit gap for each position.

    The R code constructs three polygonal “pizza slices” and tests a small
    diamond around every target against each slice.  Reproduce that geometry
    when Shapely is available, including targets lying on a slice boundary;
    retain an angular fallback for lightweight diagnostics where the optional
    geometry dependency is not installed.  Returned exit numbers are 1, 2,
    and 3 to stay compatible with the R diagnostics.
    """

    positions = _positions(positions)
    exits = constants.ceg_positions[:, :2]
    exit_degrees = np.array([-90.0, 30.0, 150.0])
    try:
        from shapely.geometry import Polygon
    except ImportError:  # pragma: no cover - the package dependency supplies it
        Polygon = None
    result: list[int] = []
    for i, point in positions.iterrows():
        counts = np.zeros(3, dtype=int)
        pizza_polygons = None
        if Polygon is not None:
            # These are the same three 120-degree sectors as the R engine.
            # Their one-degree arc sampling is only for choosing a cable exit;
            # it is deliberately separate from the five-degree probe-head
            # approximation used for physical conflict checks.
            sector_ranges = (
                np.arange(-30.0, -151.0, -1.0),
                np.arange(-30.0, 91.0, 1.0),
                np.arange(90.0, 211.0, 1.0),
            )
            pizza_polygons = [
                Polygon(
                    [(point.x, point.y)]
                    + [
                        (
                            constants.plate_radius * math.cos(math.radians(angle)),
                            constants.plate_radius * math.sin(math.radians(angle)),
                        )
                        for angle in arc
                    ]
                )
                for arc in sector_ranges
            ]
        for j, other in positions.iterrows():
            if i == j:
                continue
            if pizza_polygons is not None:
                diamond = Polygon([
                    (other.x + constants.excl_radius, other.y),
                    (other.x, other.y + constants.excl_radius),
                    (other.x - constants.excl_radius, other.y),
                    (other.x, other.y - constants.excl_radius),
                ])
                for index, pizza in enumerate(pizza_polygons):
                    if pizza.intersects(diamond):
                        counts[index] += 1
            else:
                angle = math.degrees(math.atan2(other.y - point.y, other.x - point.x))
                # Fallback approximation of the three R “pizza” sectors.
                distances = np.abs(((angle - exit_degrees + 180.0) % 360.0) - 180.0)
                counts[int(np.argmin(distances))] += 1
        candidates = np.flatnonzero(counts == counts.min())
        distances_to_exits = np.hypot(exits[candidates, 0] - point.x, exits[candidates, 1] - point.y)
        chosen = int(candidates[np.argmin(distances_to_exits)])
        # Do not choose an exit if the cable/ferrule would effectively block
        # it.  The R engine deliberately moves such a probe to the second
        # nearest exit, regardless of which exit won the crowding tie above.
        # Picking the first exit other than ``chosen`` is not equivalent: if
        # the least-crowded exit was already the second-nearest one, that
        # shortcut incorrectly selected the nearest exit.
        distance = float(np.hypot(exits[chosen, 0] - point.x, exits[chosen, 1] - point.y))
        e0 = math.sqrt((constants.wceg / 2.0) ** 2 - 9.0 ** 2)
        if distance - (constants.probe_l + constants.cable_l + constants.tip_l / 2.0) < e0:
            order = np.argsort(np.hypot(exits[:, 0] - point.x, exits[:, 1] - point.y))
            chosen = int(order[1])
        result.append(chosen + 1)  # retain R's 1, 2, 3 convention
    return np.asarray(result, dtype=int)


def first_guess_angles(positions: pd.DataFrame, cegs: Iterable[int], constants: HectorConstants, guide: bool = False) -> np.ndarray:
    """Point each probe toward its selected exit, using the legacy convention."""
    positions = _positions(positions)
    cegs = list(cegs)
    if len(cegs) != len(positions):
        raise ValueError("There must be one cable-exit choice for every probe position.")
    exits = constants.ceg_positions[:, :2]
    guesses = []
    for row, ceg in zip(positions.itertuples(), cegs):
        if int(ceg) not in (1, 2, 3):
            raise ValueError(f"Cable-exit choices must be 1, 2, or 3; got {ceg!r}.")
        exit_x, exit_y = exits[int(ceg) - 1]
        angle = math.atan2(exit_y - row.y, exit_x - row.x) + math.pi
        guesses.append(angle % (2.0 * math.pi))
    return np.asarray(guesses, dtype=float)


def angle_range(angle: float, delta_degrees: float) -> tuple[float, float]:
    """Return a symmetric local search interval around an angle in radians."""
    return angle - math.radians(delta_degrees), angle + math.radians(delta_degrees)


def _adjust_angle(theta_degrees: float, first_guess_degrees: float) -> float:
    """Choose the equivalent angle representation nearest the first guess."""
    if theta_degrees < first_guess_degrees - 180.0:
        theta_degrees += 360.0
    elif theta_degrees > first_guess_degrees + 180.0:
        theta_degrees -= 360.0
    return theta_degrees


def compute_angles_range(
    x: float,
    y: float,
    ceg: int,
    constants: HectorConstants,
    *,
    delta_ang: float = 20.0,
    guide: bool = False,
) -> tuple[float, float]:
    """Port the R ``computeAnglesRange`` physical angle limits.

    The interval is limited by three things: the cable bend radius, the
    circular plate edge, and the selected cable-exit gap.  The stochastic
    search must sample inside this interval; sampling arbitrary angles and
    rejecting them afterwards is both slower and materially different from the
    R engine.
    """

    point = pd.DataFrame({"x": [float(x)], "y": [float(y)]})
    first_guess = math.degrees(first_guess_angles(point, [ceg], constants, guide=guide)[0])
    tip_l = constants.gtip_l if guide else constants.tip_l
    probe_l = constants.gprobe_l if guide else constants.probe_l
    cable_l = constants.gcable_l if guide else constants.cable_l
    # The stable R engine uses the science-cable width for the CEG angular
    # margin even in the guide branch. Keep that distinction from the probe
    # body width so future guide constants do not silently change the port.
    ceg_cable_w = constants.cable_w
    exit_x, exit_y, exit_degree = constants.ceg_positions[int(ceg) - 1]
    distance_to_exit = math.hypot(exit_x - x, exit_y - y)
    e = distance_to_exit - (probe_l + constants.tip_l / 2.0)
    if e < 2.0 * constants.fbr:
        ratio = max(-1.0, min(1.0, e / (2.0 * constants.fbr)))
        new_delta = math.degrees(math.asin(ratio))
        if new_delta < delta_ang:
            delta_ang = new_delta

    min_angle = first_guess - delta_ang
    max_angle = first_guess + delta_ang
    radius = constants.plate_radius
    d = math.hypot(x, y)

    # Intersect the circle centred on the target with the plate circle.  These
    # intersections define the angular edge limits when the complete probe
    # could reach the plate boundary.
    if d > 0.0 and d + (cable_l + probe_l + tip_l) >= radius:
        r1 = cable_l + probe_l + tip_l
        a = (radius**2 - r1**2 + d**2) / (2.0 * d)
        h2 = radius**2 - a**2
        if h2 >= 0.0:
            h = math.sqrt(max(0.0, h2))
            x1 = a * x / d + h * y / d
            y1 = a * y / d - h * x / d
            x2 = a * x / d - h * y / d
            y2 = a * y / d + h * x / d
            edge1 = _adjust_angle(math.degrees(math.atan2(y1 - y, x1 - x)) - 180.0, first_guess)
            edge2 = _adjust_angle(math.degrees(math.atan2(y2 - y, x2 - x)) - 180.0, first_guess)
            edge_min, edge_max = min(edge1, edge2), max(edge1, edge2)

            # If the ferrule/cable passes through the selected gap, the gap
            # itself, rather than the circular edge, sets the allowed interval.
            dx = constants.wceg / 2.0 * sind(exit_degree)
            dy = -constants.wceg / 2.0 * cosd(exit_degree)
            p2 = (exit_x - dx, exit_y - dy)
            p3 = (exit_x + dx, exit_y + dy)
            d12 = math.hypot(x - p2[0], y - p2[1])
            d13 = math.hypot(x - p3[0], y - p3[1])
            dcegth = cable_l + probe_l + tip_l / 2.0
            if d12 < dcegth or d13 < dcegth:
                psi2 = math.degrees(math.asin(max(-1.0, min(1.0, ceg_cable_w / 2.0 / max(d12, 1e-12)))))
                psi3 = math.degrees(math.asin(max(-1.0, min(1.0, ceg_cable_w / 2.0 / max(d13, 1e-12)))))
                gap1 = _adjust_angle(math.degrees(math.atan2(y - p2[1], x - p2[0])) - psi2, first_guess)
                gap2 = _adjust_angle(math.degrees(math.atan2(y - p3[1], x - p3[0])) + psi3, first_guess)
                min_angle, max_angle = min(gap1, gap2), max(gap1, gap2)
            else:
                # Match the R branch conditions exactly.  These tests must
                # be based on the first-guess angle, not on the provisional
                # stochastic interval.  Using ``min_angle``/``max_angle``
                # here can clip the wrong end of an interval that crosses
                # the -180/180 representation boundary and produce
                # ``min_angle > max_angle`` for an otherwise valid probe.
                if first_guess < edge_min and first_guess < edge_max:
                    if max_angle > edge_min:
                        max_angle = edge_min
                elif first_guess > edge_min and first_guess > edge_max:
                    if min_angle < edge_max:
                        min_angle = edge_max
                elif first_guess > min_angle and first_guess < max_angle:
                    if min_angle < edge_min:
                        min_angle = edge_min
                    elif max_angle > edge_max:
                        max_angle = edge_max

    return float(min_angle), float(max_angle)


def angle_plugability_status(
    positions: pd.DataFrame,
    angles: Iterable[float],
    cegs: Iterable[int],
    constants: HectorConstants,
    *,
    guide: bool = False,
    alignment_tolerance_degrees: float = 0.5,
) -> dict[str, object]:
    """Check angle representations and alignment with the selected CEGs.

    ``compute_angles_range`` defines the physically allowed interval around a
    cable exit.  A stochastic trial can nevertheless carry an equivalent
    angle representation outside that interval (for example 403 degrees
    instead of -317 degrees), so the check tests all equivalent ``2*pi``
    representations explicitly.  The alignment values are retained as a
    soft objective: among otherwise valid configurations, a probe closer to
    its natural exit-pointing angle is preferable.
    """

    positions = _positions(positions)
    angles = np.asarray(list(angles), dtype=float)
    cegs = np.asarray(list(cegs), dtype=int)
    if len(positions) != len(angles) or len(positions) != len(cegs):
        raise ValueError("Positions, angles, and cable-exit choices must have equal length.")
    invalid: list[int] = []
    deviations: list[float] = []
    ranges: list[tuple[float, float]] = []
    for index, (row, angle, ceg) in enumerate(
        zip(positions.itertuples(), angles, cegs)
    ):
        first_guess = first_guess_angles(
            pd.DataFrame({"x": [row.x], "y": [row.y]}),
            [int(ceg)],
            constants,
            guide=guide,
        )[0]
        lower_degrees, upper_degrees = compute_angles_range(
            row.x,
            row.y,
            int(ceg),
            constants,
            delta_ang=180.0,
            guide=guide,
        )
        lower = math.radians(lower_degrees)
        upper = math.radians(upper_degrees)
        equivalent = [float(angle) + 2.0 * math.pi * shift for shift in range(-3, 4)]
        if not any(lower - 1e-12 <= candidate <= upper + 1e-12 for candidate in equivalent):
            invalid.append(index + 1)
        difference = (float(angle) - float(first_guess) + math.pi) % (2.0 * math.pi) - math.pi
        deviations.append(abs(math.degrees(difference)))
        ranges.append((lower_degrees, upper_degrees))
    misaligned = [
        index + 1
        for index, deviation in enumerate(deviations)
        if deviation > alignment_tolerance_degrees
    ]
    return {
        "ok": not invalid,
        "invalid": invalid,
        "misaligned": misaligned,
        "deviations_degrees": deviations,
        "ranges_degrees": ranges,
        "alignment_cost_radians": float(np.sum(np.deg2rad(deviations))),
    }


def infer_cegs_from_angles(
    positions: pd.DataFrame,
    angles: Iterable[float],
    constants: HectorConstants,
    *,
    guide: bool = False,
) -> np.ndarray:
    """Infer cable-exit gaps when reopening a configured CSV.

    The legacy CSV format stores positions and angles but not the internal CEG
    assignment.  Re-running :func:`choose_cegs` from positions alone can pick a
    different exit after a probe has been rotated manually.  For each probe we
    therefore choose the exit whose physically allowed angle interval contains
    the saved angle, preferring the exit whose natural pointing angle is
    closest.  If no exit can contain the angle, retain the positional choice so
    the validator can report a genuine invalid angle.
    """

    positions = _positions(positions)
    angles = np.asarray(list(angles), dtype=float)
    if len(positions) != len(angles):
        raise ValueError("Positions and angles must have equal length.")
    positional_choices = choose_cegs(positions, constants)
    inferred: list[int] = []
    for row, angle, fallback in zip(positions.itertuples(), angles, positional_choices):
        valid: list[tuple[float, int]] = []
        for ceg in (1, 2, 3):
            lower_degrees, upper_degrees = compute_angles_range(
                row.x,
                row.y,
                ceg,
                constants,
                delta_ang=180.0,
                guide=guide,
            )
            lower = math.radians(lower_degrees)
            upper = math.radians(upper_degrees)
            equivalent = [float(angle) + 2.0 * math.pi * shift for shift in range(-3, 4)]
            if not any(lower - 1e-12 <= candidate <= upper + 1e-12 for candidate in equivalent):
                continue
            first_guess = first_guess_angles(
                pd.DataFrame({"x": [row.x], "y": [row.y]}),
                [ceg],
                constants,
                guide=guide,
            )[0]
            difference = (float(angle) - float(first_guess) + math.pi) % (2.0 * math.pi) - math.pi
            valid.append((abs(difference), ceg))
        inferred.append(min(valid)[1] if valid else int(fallback))
    return np.asarray(inferred, dtype=int)


def _target_overlap_area(polygons: list[np.ndarray]) -> float:
    """Sum pairwise overlap areas, matching the R search objective."""
    total = 0.0
    for i in range(len(polygons) - 1):
        for j in range(i + 1, len(polygons)):
            total += intersection_area(polygons[i], polygons[j])
    return total


def minimize_conflicts(
    positions: pd.DataFrame,
    angles: np.ndarray,
    cegs: np.ndarray,
    constants: HectorConstants,
    conflicts: pd.DataFrame,
    *,
    delta_ang: float = 20.0,
    seed: int = 2024,
    max_recursive: int = 3,
) -> np.ndarray:
    """Port the R ``MinimizeConflicts`` stochastic angle search.

    Only conflicted probes and probes within one third of a probe length are
    varied.  Every other angle is held fixed, and the trial count is based on
    the number of movable probes as in the R implementation.
    """

    if conflicts.empty:
        return np.asarray(angles, dtype=float).copy()
    positions = _positions(positions)
    angles = np.asarray(angles, dtype=float).copy()
    cegs = np.asarray(cegs, dtype=int)
    conflicted = set((conflicts["Probe_1st"] - 1).astype(int)) | set((conflicts["Probe_2nd"] - 1).astype(int))
    movable = set(conflicted)
    for index in list(conflicted):
        distances = np.hypot(positions.x - positions.iloc[index].x, positions.y - positions.iloc[index].y)
        movable.update(np.flatnonzero(distances.to_numpy() < constants.probe_l / 3.0).tolist())
    movable = sorted(movable)
    bounds = np.column_stack([angles.copy(), angles.copy()])
    for index in movable:
        bounds[index] = np.deg2rad(compute_angles_range(
            positions.iloc[index].x,
            positions.iloc[index].y,
            int(cegs[index]),
            constants,
            delta_ang=delta_ang,
            guide=False,
        ))
    rng = np.random.default_rng(seed)
    trials = max(100, len(movable) * 500)
    best_angles = angles.copy()
    best_polygons = make_polygons(positions, best_angles, constants)
    best_area = _target_overlap_area(best_polygons)
    trial_matrix = np.tile(angles, (trials, 1))
    for index in movable:
        low, high = bounds[index]
        if low > high:
            continue
        trial_matrix[:, index] = rng.uniform(low, high, trials)
    trial_matrix[0, :] = angles
    for trial in trial_matrix:
        polygons = make_polygons(positions, trial, constants)
        # R evaluates overlap area here, then the caller performs the strict
        # complete-polygon FOV check before accepting the final field.
        area = _target_overlap_area(polygons)
        if area < best_area:
            best_area = area
            best_angles = trial.copy()
            if area <= 0.0:
                return best_angles
    if max_recursive > 0:
        residual = find_probe_conflicts(make_polygons(positions, best_angles, constants))
        # The R routine recurses whenever conflicts remain, even if the
        # number of rows is unchanged: a different conflict pair can expose
        # a different set of movable probes. Keep the retry bounded so a hard
        # field cannot create an unbounded loop.
        if not residual.empty:
            return minimize_conflicts(
                positions,
                best_angles,
                cegs,
                constants,
                residual,
                delta_ang=delta_ang,
                seed=seed + 1,
                max_recursive=max_recursive - 1,
            )
    return best_angles


def try_next_ceg(
    positions: pd.DataFrame,
    angles: np.ndarray,
    cegs: np.ndarray,
    constants: HectorConstants,
    conflicts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Port the R ``try_next_ceg`` cluster-based exit-gap swap.

    A previous Python shortcut kept a new exit only when the immediate
    conflict count decreased. The R engine deliberately changes one probe in
    every connected conflict cluster and then lets the stochastic angle search
    exploit the new geometry, even when the immediate count is unchanged.
    """

    positions = _positions(positions)
    cegs_new = np.asarray(cegs, dtype=int).copy()
    angles_new = np.asarray(angles, dtype=float).copy()
    pairs = [
        (int(row.Probe_1st) - 1, int(row.Probe_2nd) - 1)
        for row in conflicts.itertuples(index=False)
    ]
    # Connected components reproduce the R conflict-cluster construction.
    clusters: list[set[int]] = []
    for pair in pairs:
        cluster = set(pair)
        matching = [i for i, existing in enumerate(clusters) if existing & cluster]
        if matching:
            merged = set().union(cluster, *(clusters[i] for i in matching))
            clusters = [existing for i, existing in enumerate(clusters) if i not in matching]
            clusters.append(merged)
        else:
            clusters.append(cluster)

    exits = constants.ceg_positions[:, :2]
    for cluster in clusters:
        members = sorted(cluster)
        cluster_cegs = cegs_new[members]
        unique_cegs = list(dict.fromkeys(int(value) for value in cluster_cegs))
        counts = [int(np.sum(cluster_cegs == value)) for value in unique_cegs]
        if len(members) - len(unique_cegs) > 0:
            # R's Mode() returns the first tied value. Choose the probe
            # furthest from that shared exit, as the R code does.
            common_ceg = int(unique_cegs[int(np.argmax(counts))])
            candidates = [index for index in members if cegs_new[index] == common_ceg]
            distances = [
                float(np.hypot(
                    exits[common_ceg - 1, 0] - positions.iloc[index].x,
                    exits[common_ceg - 1, 1] - positions.iloc[index].y,
                ))
                for index in candidates
            ]
            swap_index = candidates[int(np.argmax(distances))]
        else:
            widths = []
            for index in members:
                low, high = compute_angles_range(
                    positions.iloc[index].x,
                    positions.iloc[index].y,
                    int(cegs_new[index]),
                    constants,
                )
                widths.append(abs(high - low))
            swap_index = members[int(np.argmin(widths))]

        exit_distances = np.hypot(
            exits[:, 0] - positions.iloc[swap_index].x,
            exits[:, 1] - positions.iloc[swap_index].y,
        )
        order = np.argsort(exit_distances)
        nearest, second = int(order[0] + 1), int(order[1] + 1)
        if (
            cegs_new[swap_index] == second
            and exit_distances[order[0]] > constants.probe_l + constants.tip_l + constants.cable_l
        ):
            cegs_new[swap_index] = nearest
        else:
            cegs_new[swap_index] = second
        # The R implementation resets all first guesses after each cluster.
        angles_new = first_guess_angles(positions, cegs_new, constants)
        angles_new[angles_new > 1.5 * math.pi] -= 2.0 * math.pi

    return angles_new, cegs_new


def wrap_minimize_conflicts(
    positions: pd.DataFrame,
    angles: np.ndarray,
    cegs: np.ndarray,
    constants: HectorConstants,
    conflicts: pd.DataFrame,
    *,
    seed: int = 2024,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Port the R wrapper that retries searches and swaps cable exits."""

    current_angles = np.asarray(angles, dtype=float).copy()
    current_cegs = np.asarray(cegs, dtype=int).copy()
    current_conflicts = conflicts.copy()
    initial_count = len(current_conflicts)
    for cycle in range(3):
        for delta in (20.0, 45.0, 90.0):
            trial = minimize_conflicts(
                positions,
                current_angles,
                current_cegs,
                constants,
                current_conflicts,
                delta_ang=delta,
                seed=seed + cycle * 100 + int(delta),
            )
            trial_conflicts = find_probe_conflicts(make_polygons(positions, trial, constants))
            # R accepts a changed stochastic result even when the number of
            # conflict rows is unchanged; the overlap area has still improved
            # and the next pass may expose a different movable set.
            if not np.array_equal(trial, current_angles) or len(trial_conflicts) < len(current_conflicts):
                current_angles, current_conflicts = trial, trial_conflicts
            if current_conflicts.empty:
                return current_angles, current_cegs, {"resolved": True, "initial_conflicts": initial_count, "cycles": cycle + 1}
        if current_conflicts.empty:
            break
        swapped_angles, swapped_cegs = try_next_ceg(
            positions, current_angles, current_cegs, constants, current_conflicts
        )
        swapped_conflicts = find_probe_conflicts(make_polygons(positions, swapped_angles, constants))
        # Preserve the R behaviour even when the immediate conflict count is
        # unchanged: the following angle search may benefit from the new exit
        # gap. The outer loop is bounded to three cycles.
        if not np.array_equal(swapped_cegs, current_cegs) or len(swapped_conflicts) < len(current_conflicts):
            current_angles, current_cegs, current_conflicts = swapped_angles, swapped_cegs, swapped_conflicts
        else:
            break
    return current_angles, current_cegs, {
        "resolved": current_conflicts.empty,
        "initial_conflicts": initial_count,
        "remaining_conflicts": len(current_conflicts),
    }


def _score(
    target_positions: pd.DataFrame,
    target_angles: np.ndarray,
    guide_positions: pd.DataFrame | None,
    guide_angles: np.ndarray | None,
    constants: HectorConstants,
    *,
    target_cegs: Iterable[int] | None = None,
    guide_cegs: Iterable[int] | None = None,
    include_plugability: bool = False,
) -> tuple[tuple[int, float, int, int, int, float], object, object, list[np.ndarray], list[np.ndarray], dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    """Evaluate conflicts, overlap area, and field-of-view violations.

    The tuple is ordered so the search always prefers fewer clashes first,
    then less overlap, then fewer out-of-field polygons, wedge violations,
    invalid CEG-angle representations, and finally angular distance from the
    natural exit-pointing angle. Plugability terms are included only for the
    joint/physical search so the established staged search remains comparable
    with its earlier behaviour.
    """
    target_polygons = make_polygons(target_positions, target_angles, constants)
    target_conflicts = find_probe_conflicts(target_polygons)
    guide_polygons: list[np.ndarray] = []
    guide_conflicts = pd.DataFrame()
    if guide_positions is not None and guide_angles is not None:
        guide_polygons = make_polygons(guide_positions, guide_angles, constants, guide=True)
        guide_conflicts = find_guide_conflicts(target_polygons, guide_polygons)
    total_area = 0.0
    if len(target_conflicts):
        total_area += float(target_conflicts["PercentArea"].sum())
    if len(guide_conflicts):
        total_area += float(guide_conflicts["PercentArea"].sum())
    target_fov = fov_status(target_polygons, constants.plate_radius)
    guide_fov = fov_status(guide_polygons, constants.plate_radius)
    if include_plugability:
        target_cegs = (
            choose_cegs(target_positions, constants)
            if target_cegs is None
            else np.asarray(list(target_cegs), dtype=int)
        )
        guide_cegs = (
            None
            if guide_positions is None
            else (
                choose_cegs(guide_positions, constants)
                if guide_cegs is None
                else np.asarray(list(guide_cegs), dtype=int)
            )
        )
        target_angle_status = angle_plugability_status(
            target_positions, target_angles, target_cegs, constants
        )
        guide_angle_status = (
            {"ok": True, "invalid": [], "misaligned": [], "deviations_degrees": [], "ranges_degrees": [], "alignment_cost_radians": 0.0}
            if guide_positions is None
            else angle_plugability_status(
                guide_positions,
                [] if guide_angles is None else guide_angles,
                [] if guide_cegs is None else guide_cegs,
                constants,
                guide=True,
            )
        )
        wedges = wedge_status(
            target_positions,
            target_angles,
            guide_positions,
            guide_angles,
            constants,
        )
        wedge_count = len(wedges["violations"])
        invalid_count = len(target_angle_status["invalid"]) + len(guide_angle_status["invalid"])
        alignment_cost = (
            float(target_angle_status["alignment_cost_radians"])
            + float(guide_angle_status["alignment_cost_radians"])
        )
    else:
        target_angle_status = {"ok": True, "invalid": [], "misaligned": [], "deviations_degrees": [], "ranges_degrees": [], "alignment_cost_radians": 0.0}
        guide_angle_status = {"ok": True, "invalid": [], "misaligned": [], "deviations_degrees": [], "ranges_degrees": [], "alignment_cost_radians": 0.0}
        wedges = {"ok": True, "pinches": [], "violations": []}
        wedge_count = invalid_count = 0
        alignment_cost = 0.0
    outside = len(target_fov["outside"]) + len(guide_fov["outside"])
    conflicts = len(target_conflicts) + len(guide_conflicts)
    return (
        (conflicts, total_area, outside, wedge_count, invalid_count, alignment_cost),
        target_conflicts,
        guide_conflicts,
        target_polygons,
        guide_polygons,
        target_fov,
        guide_fov,
        wedges,
        target_angle_status,
        guide_angle_status,
    )


def _better(first: tuple[int, float, int], second: tuple[int, float, int]) -> bool:
    """Return whether the first search score is preferable to the second."""
    return first < second


def _flip_candidates(angle: float) -> list[float]:
    """Return equivalent representations of the 180-degree alternative."""
    return [angle + math.pi + 2.0 * math.pi * offset for offset in (-1, 0, 1)]


def _valid_flip(
    angle: float,
    x: float,
    y: float,
    ceg: int,
    constants: HectorConstants,
    *,
    guide: bool,
) -> float | None:
    """Return the 180-degree alternative if it lies in the R angle range."""
    low, high = np.deg2rad(compute_angles_range(
        x, y, ceg, constants, delta_ang=180.0, guide=guide
    ))
    candidates = [candidate for candidate in _flip_candidates(angle) if low <= candidate <= high]
    return candidates[0] if candidates else None


def joint_search(
    target_positions: pd.DataFrame,
    target_angles: np.ndarray,
    guide_positions: pd.DataFrame | None,
    guide_angles: np.ndarray | None,
    target_indices: Iterable[int],
    guide_indices: Iterable[int],
    constants: HectorConstants,
    *,
    seed: int = 2024,
    deltas: Iterable[float] = (20.0, 45.0, 90.0),
    samples: int = 240,
    target_cegs: Iterable[int] | None = None,
    guide_cegs: Iterable[int] | None = None,
    include_plugability: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    """Search conflicted target and guide angles together.

    The search starts with the current angles, tries 180-degree alternatives,
    and then samples small stochastic tweaks around both probe types.  The
    target and guide index lists are explicit: this is what allows a
    target-guide clash to move *both* probes during a trial while leaving the
    rest of the field fixed.  A failed trial never mutates the caller's
    arrays.  The returned diagnostics make it clear whether the search
    actually resolved the conflict.
    """

    targets = _positions(target_positions)
    guides = None if guide_positions is None else _positions(guide_positions)
    target_cegs = np.asarray(
        list(target_cegs) if target_cegs is not None else choose_cegs(targets, constants),
        dtype=int,
    )
    guide_cegs = None if guides is None else np.asarray(
        list(guide_cegs) if guide_cegs is not None else choose_cegs(guides, constants),
        dtype=int,
    )
    base_targets = np.asarray(target_angles, dtype=float).copy()
    base_guides = None if guide_angles is None else np.asarray(guide_angles, dtype=float).copy()
    target_indices = sorted(set(int(i) for i in target_indices))
    guide_indices = sorted(set(int(i) for i in guide_indices))
    rng = np.random.default_rng(seed)

    best_targets, best_guides = base_targets.copy(), None if base_guides is None else base_guides.copy()
    best = _score(
        targets,
        best_targets,
        guides,
        best_guides,
        constants,
        target_cegs=target_cegs,
        guide_cegs=guide_cegs,
        include_plugability=include_plugability,
    )
    best_score = best[0]

    def solved(result) -> bool:
        """Return whether a trial is safe enough to stop this local search."""
        if result[0][0] != 0 or result[0][2] != 0:
            return False
        if not include_plugability:
            return True
        return (
            result[0][3] == 0
            and result[0][4] == 0
            and result[7].get("ok", True)
            and not result[8].get("misaligned", [])
            and not result[9].get("misaligned", [])
        )

    # Calculate the physically allowed interval around each current angle.
    # The R search intersects this with a local +/- delta interval for every
    # retry, rather than sampling arbitrary rotations and hoping the FOV test
    # rejects them later.
    def local_bounds(kind: str, index: int, delta: float, base: float) -> tuple[float, float] | None:
        if kind == "target":
            row = targets.iloc[index]
            ceg = int(target_cegs[index])
            guide = False
        else:
            if guides is None:
                return None
            row = guides.iloc[index]
            ceg = int(guide_cegs[index])
            guide = True
        allowed_low, allowed_high = np.deg2rad(compute_angles_range(
            row.x, row.y, ceg, constants, delta_ang=180.0, guide=guide
        ))
        # Angles are periodic, but the physical interval returned by the R
        # routine is expressed near the natural first guess.  Project an
        # invalid equivalent representation back onto that interval before
        # applying the local stochastic width.  Without this, an angle such
        # as 403 degrees could be retained unchanged because its numeric
        # representation lay outside the sampled interval even though the
        # corresponding probe orientation should have been repaired.
        allowed_candidates = [
            float(base) + 2.0 * math.pi * shift for shift in range(-3, 4)
        ]
        valid_candidates = [
            candidate for candidate in allowed_candidates
            if allowed_low <= candidate <= allowed_high
        ]
        if valid_candidates:
            anchor = min(valid_candidates, key=lambda candidate: abs(candidate - base))
        else:
            guess = first_guess_angles(
                pd.DataFrame({"x": [row.x], "y": [row.y]}),
                [ceg],
                constants,
                guide=guide,
            )[0]
            guess_candidates = [
                float(guess) + 2.0 * math.pi * shift for shift in range(-3, 4)
            ]
            valid_guesses = [
                candidate for candidate in guess_candidates
                if allowed_low <= candidate <= allowed_high
            ]
            anchor = valid_guesses[0] if valid_guesses else float(np.clip(guess, allowed_low, allowed_high))
        low = max(allowed_low, anchor - math.radians(delta))
        high = min(allowed_high, anchor + math.radians(delta))
        return (low, high) if low <= high else None

    # A low-dimensional flip neighbourhood is cheap and often finds the
    # physically obvious solution before random sampling begins.  For two
    # conflicted probes this checks the current angle, target flip, guide flip,
    # and both flipped before spending time on random tweaks.
    movable = [("target", index) for index in target_indices]
    movable.extend(("guide", index) for index in guide_indices)
    if len(movable) <= 6:
        masks = range(1, 2 ** len(movable))
        for mask in masks:
            trial_targets = base_targets.copy()
            trial_guides = None if base_guides is None else base_guides.copy()
            for bit, (kind, index) in enumerate(movable):
                if mask & (1 << bit):
                    if kind == "target":
                        row = targets.iloc[index]
                        ceg = int(target_cegs[index])
                        flipped = _valid_flip(base_targets[index], row.x, row.y, ceg, constants, guide=False)
                        if flipped is None:
                            continue
                        trial_targets[index] = flipped
                    elif trial_guides is not None:
                        row = guides.iloc[index]
                        ceg = int(guide_cegs[index])
                        flipped = _valid_flip(base_guides[index], row.x, row.y, ceg, constants, guide=True)
                        if flipped is None:
                            continue
                        trial_guides[index] = flipped
            result = _score(
                targets,
                trial_targets,
                guides,
                trial_guides,
                constants,
                target_cegs=target_cegs,
                guide_cegs=guide_cegs,
                include_plugability=include_plugability,
            )
            if _better(result[0], best_score):
                best_score = result[0]
                best_targets, best_guides = trial_targets, trial_guides
            if solved(result):
                return trial_targets, trial_guides, {"resolved": True, "trials": mask, "score": result[0]}

    for delta in deltas:
        for trial_number in range(max(1, samples)):
            trial_targets = base_targets.copy()
            trial_guides = None if base_guides is None else base_guides.copy()
            for index in target_indices:
                bounds = local_bounds("target", index, delta, base_targets[index])
                if bounds is None:
                    continue
                if trial_number % 7 == 0:
                    row = targets.iloc[index]
                    ceg = int(target_cegs[index])
                    flipped = _valid_flip(base_targets[index], row.x, row.y, ceg, constants, guide=False)
                    # A 180-degree alternative is only a starting point: the
                    # useful solution is often the flip plus a small tweak.
                    # Recompute the local interval around the flip and keep
                    # every draw inside the physical R angle range.
                    flip_bounds = None if flipped is None else local_bounds("target", index, delta, flipped)
                    trial_targets[index] = rng.uniform(*(flip_bounds or bounds))
                else:
                    trial_targets[index] = rng.uniform(*bounds)
            if trial_guides is not None:
                for index in guide_indices:
                    bounds = local_bounds("guide", index, delta, base_guides[index])
                    if bounds is None:
                        continue
                    if trial_number % 7 == 0:
                        row = guides.iloc[index]
                        ceg = int(guide_cegs[index])
                        flipped = _valid_flip(base_guides[index], row.x, row.y, ceg, constants, guide=True)
                        flip_bounds = None if flipped is None else local_bounds("guide", index, delta, flipped)
                        trial_guides[index] = rng.uniform(*(flip_bounds or bounds))
                    else:
                        trial_guides[index] = rng.uniform(*bounds)
            result = _score(
                targets,
                trial_targets,
                guides,
                trial_guides,
                constants,
                target_cegs=target_cegs,
                guide_cegs=guide_cegs,
                include_plugability=include_plugability,
            )
            if _better(result[0], best_score):
                best_score = result[0]
                best_targets, best_guides = trial_targets, trial_guides
            if solved(result):
                return trial_targets, trial_guides, {"resolved": True, "trials": trial_number + 1, "delta": delta, "score": result[0]}

    return best_targets, best_guides, {"resolved": best_score[0] == 0 and best_score[2] == 0, "trials": samples, "score": best_score}


def minimize_guide_conflicts(
    target_positions: pd.DataFrame,
    target_angles: np.ndarray,
    target_cegs: Iterable[int],
    guide_positions: pd.DataFrame,
    guide_angles: np.ndarray,
    guide_cegs: Iterable[int],
    constants: HectorConstants,
    guide_conflicts: pd.DataFrame,
    *,
    seed: int = 2024,
    extended_samples_per_probe: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Port the R guide solver, including guide-only and joint searches.

    The R engine uses a small search first and reserves a larger
    ``samples_per_probe=400`` search for genuinely difficult target-guide
    clashes.  The earlier Python port stopped after the small search, which
    made it much more likely to retain a solvable guide for manual repair.
    The larger budget is now used as the final bounded attempt.  It can be
    reduced for exploratory runs, but values below 1 are rejected.
    """

    if extended_samples_per_probe < 1:
        raise ValueError("extended_samples_per_probe must be at least 1.")

    target_positions = _positions(target_positions)
    guide_positions = _positions(guide_positions)
    target_angles = np.asarray(target_angles, dtype=float).copy()
    guide_angles = np.asarray(guide_angles, dtype=float).copy()
    target_cegs = np.asarray(list(target_cegs), dtype=int)
    guide_cegs = np.asarray(list(guide_cegs), dtype=int)
    guide_guide = guide_conflicts["ConflictType"].eq("guide-guide")
    guide_indices = set((guide_conflicts.loc[~guide_guide, "Probe_2nd"] - 1).astype(int))
    guide_indices.update((guide_conflicts.loc[guide_guide, "Probe_1st"] - 1).astype(int))
    guide_indices.update((guide_conflicts.loc[guide_guide, "Probe_2nd"] - 1).astype(int))
    target_indices = set((guide_conflicts.loc[~guide_guide, "Probe_1st"] - 1).astype(int))
    guide_indices = sorted(index for index in guide_indices if 0 <= index < len(guide_positions))
    target_indices = sorted(index for index in target_indices if 0 <= index < len(target_positions))

    n_joint = len(target_indices) + len(guide_indices)

    # The stable R engine first tries the implicated target and guide together
    # with a modest neighbourhood.  This is important: a target-guide clash
    # often needs the target to flip while the guide makes a small adjustment.
    # The previous Python version tried guide-only first, which could consume
    # time without ever exploring that obvious two-probe solution.
    early_joint = joint_search(
        target_positions,
        target_angles,
        guide_positions,
        guide_angles,
        target_indices,
        guide_indices,
        constants,
        seed=seed,
        deltas=(10.0, 20.0, 45.0),
        samples=max(100, n_joint * 50),
        target_cegs=target_cegs,
        guide_cegs=guide_cegs,
    ) if target_indices else None
    if early_joint is not None and early_joint[2].get("resolved"):
        return early_joint[0], early_joint[1], guide_cegs, {"mode": "joint-early", **early_joint[2]}

    # Keep the guide-only pass because some clashes genuinely can be fixed by
    # rotating only the new guide, and it is cheaper than moving a target.
    guide_only = joint_search(
        target_positions,
        target_angles,
        guide_positions,
        guide_angles,
        [],
        guide_indices,
        constants,
        seed=seed,
        deltas=(20.0, 45.0, 90.0),
        samples=max(100, len(guide_indices) * 50),
        target_cegs=target_cegs,
        guide_cegs=guide_cegs,
    )
    if guide_only[2].get("resolved"):
        return guide_only[0], guide_only[1], guide_cegs, {"mode": "guide-only", **guide_only[2]}

    # Last chance for a hard field: increase the number of random draws per
    # implicated probe and include both small and large angular neighbourhoods.
    # This is deliberately bounded and only varies the probes involved in the
    # current conflict, so the extra effort is spent where it can help.
    if target_indices:
        extended_joint = joint_search(
            target_positions,
            target_angles,
            guide_positions,
            guide_angles,
            target_indices,
            guide_indices,
            constants,
            seed=seed + 1000,
            deltas=(5.0, 10.0, 20.0, 45.0, 90.0),
            samples=max(100, n_joint * extended_samples_per_probe),
            target_cegs=target_cegs,
            guide_cegs=guide_cegs,
        )
        if extended_joint[2].get("resolved"):
            return extended_joint[0], extended_joint[1], guide_cegs, {"mode": "joint-extended", **extended_joint[2]}
        return extended_joint[0], extended_joint[1], guide_cegs, {"mode": "joint-extended-unsolved", **extended_joint[2]}
    return guide_only[0], guide_only[1], guide_cegs, {"mode": "guide-only-unsolved", **guide_only[2]}


def optimize_targets(
    positions: pd.DataFrame,
    angles: np.ndarray,
    constants: HectorConstants,
    *,
    cegs: Iterable[int] | None = None,
    seed: int = 2024,
) -> tuple[np.ndarray, dict[str, object]]:
    """Optimise only the initially conflicted science/standard probes."""
    positions = _positions(positions)
    cegs = choose_cegs(positions, constants) if cegs is None else np.asarray(list(cegs), dtype=int)
    polygons = make_polygons(positions, angles, constants)
    conflicts = find_probe_conflicts(polygons)
    if conflicts.empty:
        return np.asarray(angles, dtype=float), {"resolved": True, "initial_conflicts": 0}
    result_targets, result_cegs, diagnostics = wrap_minimize_conflicts(
        positions,
        np.asarray(angles),
        cegs,
        constants,
        conflicts,
        seed=seed,
    )
    return result_targets, {"initial_conflicts": len(conflicts), "cegs": result_cegs, **diagnostics}


@dataclass
class SearchState:
    """Convenient container for a fully evaluated trial configuration."""
    target_positions: pd.DataFrame
    target_angles: np.ndarray
    guide_positions: pd.DataFrame
    guide_angles: np.ndarray
    target_conflicts: pd.DataFrame
    guide_conflicts: pd.DataFrame
    target_fov: dict[str, object]
    guide_fov: dict[str, object]
