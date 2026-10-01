"""Public configuration workflow."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .constants import HectorConstants, load_constants
from .engine import (
    _score,
    angle_plugability_status,
    choose_cegs,
    first_guess_angles,
    joint_search,
    minimize_guide_conflicts,
    optimize_targets,
)
from .geometry import (
    find_guide_conflicts,
    find_probe_conflicts,
    fov_status,
    make_polygons,
    point_to_polygon_distances,
    wedge_status,
)
from .io import build_output_tables, hector_read_tables, hector_write_tables, prepare_field
from .plotting import plot_configured_field


@dataclass
class ConfigurationResult:
    """Result returned by :func:`configure_hector`.

    The attributes are deliberately ordinary pandas/numpy objects so that an
    R user can inspect them in a familiar way.  ``config`` contains the
    compact diagnostic dictionary; ``hexas`` and ``guides`` are the tables
    ready for the robot pipeline or for manual repair in the app.
    """

    config: dict[str, Any]
    hexas: pd.DataFrame
    guides: pd.DataFrame
    inputs: Any
    constants: HectorConstants

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def refresh_result_tables(result: ConfigurationResult) -> ConfigurationResult:
    """Rebuild output tables after a manual angle edit in the app.

    The automatic solver stores positions/angles in ``result.config`` while
    the downloadable CSVs are separate DataFrames.  Keeping this small
    synchronisation step explicit prevents the plot from showing a repaired
    probe while the downloaded file still contains its old angle.
    """

    target_positions = result.config["pos"].reset_index(drop=True)
    guide_positions = result.config["gpos"].reset_index(drop=True)
    internal = SimpleNamespace(
        target_ids=list(result.config["pos"].index.astype(str)),
        target_positions=target_positions,
        target_angles=np.asarray(result.config["angs"]),
        guide_ids=list(result.config["gpos"].index.astype(str)),
        guide_positions=guide_positions,
        guide_angles=np.asarray(result.config["gangs"]),
    )
    outputs = build_output_tables(result.inputs, internal, result.constants)
    result.hexas = outputs["hexas"]
    result.guides = outputs["guides"]
    return result


def _positions_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Extract a clean zero-based ``x, y`` position table from input rows."""
    return rows[["x", "y"]].reset_index(drop=True).astype(float)


def _ids_from_rows(rows: pd.DataFrame) -> list[str]:
    """Extract stable string IDs used to join results back to CSV metadata."""
    return rows["ID"].astype(str).tolist()


def _clean_standard_candidates(
    rows: pd.DataFrame,
    occupied_positions: pd.DataFrame,
    occupied_angles: np.ndarray,
    constants: HectorConstants,
) -> pd.DataFrame:
    """Reproduce R ``cleanconflictedstds`` candidate ordering and filtering."""
    if rows.empty:
        return rows.copy()
    occupied_polygons = make_polygons(occupied_positions, occupied_angles, constants)
    distances = point_to_polygon_distances(rows, occupied_polygons)
    cleaned = rows.assign(_distance=distances)
    cleaned = cleaned.loc[cleaned["_distance"] > constants.excl_radius].copy()
    cleaned = cleaned.loc[
        np.hypot(cleaned["x"], cleaned["y"])
        < constants.plate_radius - 1.5 * constants.excl_radius
    ]
    cleaned["_radius"] = np.hypot(cleaned["x"], cleaned["y"])
    return (
        cleaned.sort_values("_radius", kind="stable")
        .drop(columns=["_distance", "_radius"])
        .reset_index(drop=True)
    )


def _clean_guide_candidates(
    rows: pd.DataFrame,
    occupied_positions: pd.DataFrame,
    occupied_angles: np.ndarray,
    constants: HectorConstants,
) -> pd.DataFrame:
    """Reproduce R ``cleanconflictedguides`` ordering and filtering.

    Guides farthest from the nearest occupied probe are preferred first, then
    the R quadrant round-robin ordering spreads candidates around the plate.
    Candidates removed by this initial positional filter are added back by
    :func:`configure_hector` so they remain available for angular rescue or
    manual intervention.
    """
    if rows.empty:
        return rows.copy()
    occupied_polygons = make_polygons(occupied_positions, occupied_angles, constants)
    distances = point_to_polygon_distances(rows, occupied_polygons)
    cleaned = rows.assign(_distance=distances)
    cleaned = cleaned.loc[cleaned["_distance"] > constants.gexcl_radius].copy()
    cleaned = cleaned.loc[
        np.hypot(cleaned["x"], cleaned["y"])
        < constants.plate_radius - 1.5 * constants.excl_radius
    ]
    cleaned["_radius"] = np.hypot(cleaned["x"], cleaned["y"])
    cleaned = cleaned.sort_values("_radius", ascending=False, kind="stable")
    quadrants = [
        cleaned.loc[(cleaned["x"] < 0) & (cleaned["y"] < 0)],
        cleaned.loc[(cleaned["x"] < 0) & (cleaned["y"] >= 0)],
        cleaned.loc[(cleaned["x"] >= 0) & (cleaned["y"] >= 0)],
        cleaned.loc[(cleaned["x"] >= 0) & (cleaned["y"] < 0)],
    ]
    ordered_parts: list[pd.DataFrame] = []
    for index in range(max((len(part) for part in quadrants), default=0)):
        for quadrant in quadrants:
            if index < len(quadrant):
                ordered_parts.append(quadrant.iloc[[index]])
    if not ordered_parts:
        return cleaned.drop(columns=["_distance", "_radius"]).reset_index(drop=True)
    return pd.concat(ordered_parts, ignore_index=True).drop(columns=["_distance", "_radius"])


def _guide_trial(
    target_positions: pd.DataFrame,
    target_angles: np.ndarray,
    guide_positions: pd.DataFrame,
    guide_angles: np.ndarray,
    constants: HectorConstants,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    """Evaluate one guide candidate against the current target configuration."""
    target_polygons = make_polygons(target_positions, target_angles, constants)
    guide_polygons = make_polygons(guide_positions, guide_angles, constants, guide=True)
    target_conflicts = find_probe_conflicts(target_polygons)
    guide_conflicts = find_guide_conflicts(target_polygons, guide_polygons)
    target_fov = fov_status(target_polygons, constants.plate_radius)
    guide_fov = fov_status(guide_polygons, constants.plate_radius)
    return (
        target_positions,
        np.asarray(target_angles),
        np.asarray(guide_angles),
        target_conflicts,
        guide_conflicts,
        target_fov,
        guide_fov,
    )


def _joint_candidate_pool(
    rows: pd.DataFrame,
    cleaned_rows: pd.DataFrame,
    *,
    minimum: int,
    limit: int,
) -> pd.DataFrame:
    """Build a small deterministic pool for the joint solver.

    The pool starts with candidates that pass the cheap positional cleaning
    step, then appends candidates rejected by that preliminary test.  The
    latter are deliberately kept: a different angle can sometimes rescue a
    candidate, and the point of this experiment is to test that possibility
    without committing to a huge combinatorial search over the whole input
    catalogue.
    """

    if rows.empty:
        return rows.copy().reset_index(drop=True)
    cleaned_ids = set(cleaned_rows["ID"].astype(str)) if not cleaned_rows.empty else set()
    remainder = rows.loc[~rows["ID"].astype(str).isin(cleaned_ids)]
    pool = pd.concat([cleaned_rows, remainder], ignore_index=True)
    pool = pool.drop_duplicates(subset=["ID"], keep="first")
    return pool.head(max(int(minimum), int(limit))).reset_index(drop=True)


def _select_joint_standards(
    target_positions: pd.DataFrame,
    target_angles: np.ndarray,
    target_cegs: np.ndarray,
    standard_rows: pd.DataFrame,
    constants: HectorConstants,
    *,
    candidate_pool: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, tuple[int, float, int]]:
    """Choose the standard pair before the joint angle search begins.

    This is a deliberately small beam over standard *identity*, not a second
    full solver.  Each pair is scored using its natural cable-exit angles;
    the selected pair is then allowed to move with every target and guide in
    :func:`_joint_optimize_configuration`.  Testing a few plausible pairs is
    important because choosing the first two standards can otherwise make a
    later guide conflict look impossible.
    """

    if standard_rows.empty or constants.nstdprobes == 0:
        empty = standard_rows.iloc[0:0].copy()
        return (
            empty,
            target_positions.copy(),
            np.asarray(target_angles, dtype=float).copy(),
            np.asarray(target_cegs, dtype=int).copy(),
            (0, 0.0, 0),
        )

    pool = standard_rows.head(max(constants.nstdprobes, candidate_pool)).reset_index(drop=True)
    n_required = min(constants.nstdprobes, len(pool))
    combinations_to_test = list(combinations(range(len(pool)), n_required))
    # Keep the experiment bounded even if a user supplies a very large pool.
    combinations_to_test = combinations_to_test[: max(1, candidate_pool * candidate_pool)]

    best: tuple[tuple[int, float, int], tuple[int, ...], pd.DataFrame, np.ndarray, np.ndarray] | None = None
    for combination in combinations_to_test:
        positions = target_positions.copy().reset_index(drop=True)
        angles = np.asarray(target_angles, dtype=float).copy()
        cegs = np.asarray(target_cegs, dtype=int).copy()
        for index in combination:
            candidate = pool.loc[[index], ["x", "y"]].reset_index(drop=True)
            positions = pd.concat([positions, candidate], ignore_index=True)
            new_ceg = int(choose_cegs(positions, constants)[-1])
            cegs = np.append(cegs, new_ceg)
            angles = np.append(
                angles,
                first_guess_angles(candidate, [new_ceg], constants)[0],
            )
        score = _score(positions, angles, None, None, constants)[0]
        if best is None or score < best[0]:
            best = (score, combination, positions, angles, cegs)

    assert best is not None  # ``pool`` contains at least one row here.
    selected = pool.iloc[list(best[1])].reset_index(drop=True)
    return selected, best[2], best[3], best[4], best[0]


def _select_joint_guides(
    target_positions: pd.DataFrame,
    target_angles: np.ndarray,
    guide_rows: pd.DataFrame,
    constants: HectorConstants,
    *,
    candidate_pool: int,
    beam_width: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, tuple[int, float, int]]:
    """Select six guide identities with a bounded initial beam search.

    Guide identities are the genuinely combinatorial part of the problem.
    A beam is a useful compromise here: it tests several alternatives at
    every slot, but does not enumerate every six-guide combination.  Once
    selected, all guide and target/standard angles are optimised together.
    """

    cleaned = _clean_guide_candidates(
        guide_rows, target_positions, target_angles, constants
    )
    pool = _joint_candidate_pool(
        guide_rows,
        cleaned,
        minimum=constants.ngprobesmax,
        limit=candidate_pool,
    )
    n_required = min(constants.ngprobesmax, len(pool))
    empty = pd.DataFrame(columns=["x", "y"], dtype=float)
    states: list[dict[str, Any]] = [{
        "selected": tuple(),
        "positions": empty,
        "angles": np.asarray([], dtype=float),
        "cegs": np.asarray([], dtype=int),
        "score": _score(target_positions, target_angles, empty, np.asarray([]), constants)[0],
    }]

    for _slot in range(n_required):
        expanded: list[dict[str, Any]] = []
        for state in states:
            selected_indices = set(state["selected"])
            for index in range(len(pool)):
                if index in selected_indices:
                    continue
                candidate = pool.loc[[index], ["x", "y"]].reset_index(drop=True)
                positions = pd.concat([state["positions"], candidate], ignore_index=True)
                cegs = choose_cegs(positions, constants)
                angles = first_guess_angles(positions, cegs, constants, guide=True)
                score = _score(target_positions, target_angles, positions, angles, constants)[0]
                expanded.append({
                    "selected": tuple(sorted((*state["selected"], index))),
                    "positions": positions,
                    "angles": angles,
                    "cegs": np.asarray(cegs, dtype=int),
                    "score": score,
                })
        expanded.sort(key=lambda state: (state["score"], state["selected"]))
        states = expanded[: max(1, int(beam_width))]
        if not states:
            break

    if not states:
        return empty, np.asarray([]), np.asarray([], dtype=int), (0, 0.0, 0)
    best = min(states, key=lambda state: (state["score"], state["selected"]))
    selected = pool.iloc[list(best["selected"])].reset_index(drop=True)
    return selected, best["angles"], best["cegs"], best["score"]


def _joint_conflict_components(
    target_conflicts: pd.DataFrame,
    guide_conflicts: pd.DataFrame,
    target_fov: dict[str, object],
    guide_fov: dict[str, object],
    target_angle_status: dict[str, object] | None = None,
    guide_angle_status: dict[str, object] | None = None,
    wedges: dict[str, object] | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Return connected target/guide conflict components for local searches."""

    nodes: set[tuple[str, int]] = set()
    edges: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for row in target_conflicts.itertuples(index=False):
        first = ("target", int(row.Probe_1st) - 1)
        second = ("target", int(row.Probe_2nd) - 1)
        nodes.update((first, second))
        edges.append((first, second))
    for row in guide_conflicts.itertuples(index=False):
        conflict_type = getattr(row, "ConflictType", "target-guide")
        if conflict_type == "guide-guide":
            first = ("guide", int(row.Probe_1st) - 1)
            second = ("guide", int(row.Probe_2nd) - 1)
        else:
            first = ("target", int(row.Probe_1st) - 1)
            second = ("guide", int(row.Probe_2nd) - 1)
        nodes.update((first, second))
        edges.append((first, second))
    for index in target_fov.get("outside", []):
        nodes.add(("target", int(index) - 1))
    for index in guide_fov.get("outside", []):
        nodes.add(("guide", int(index) - 1))
    for index in (target_angle_status or {}).get("invalid", []):
        nodes.add(("target", int(index) - 1))
    for index in (target_angle_status or {}).get("misaligned", []):
        nodes.add(("target", int(index) - 1))
    for index in (guide_angle_status or {}).get("invalid", []):
        nodes.add(("guide", int(index) - 1))
    for index in (guide_angle_status or {}).get("misaligned", []):
        nodes.add(("guide", int(index) - 1))
    for violation in (wedges or {}).get("violations", []):
        nodes.update(tuple(node) for node in violation.get("nodes", []))

    graph = {node: set() for node in nodes}
    for first, second in edges:
        graph[first].add(second)
        graph[second].add(first)
    components: list[tuple[list[int], list[int]]] = []
    unseen = set(nodes)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[tuple[str, int]] = set()
        while stack:
            node = stack.pop()
            if node not in unseen:
                continue
            unseen.remove(node)
            component.add(node)
            stack.extend(graph[node])
        components.append(
            (
                sorted(index for kind, index in component if kind == "target"),
                sorted(index for kind, index in component if kind == "guide"),
            )
        )
    return components


def _joint_optimize_configuration(
    target_positions: pd.DataFrame,
    target_angles: np.ndarray,
    target_cegs: np.ndarray,
    guide_positions: pd.DataFrame,
    guide_angles: np.ndarray,
    guide_cegs: np.ndarray,
    constants: HectorConstants,
    *,
    seed: int,
    samples_per_probe: int,
    max_rounds: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Optimise all selected probe classes through shared conflict components."""

    target_angles = np.asarray(target_angles, dtype=float).copy()
    guide_angles = np.asarray(guide_angles, dtype=float).copy()
    target_cegs = np.asarray(target_cegs, dtype=int)
    guide_cegs = np.asarray(guide_cegs, dtype=int)
    diagnostics: dict[str, Any] = {
        "mode": "joint-components",
        "rounds": 0,
        "components": 0,
        "trials": 0,
    }

    def resolved(score_result) -> bool:
        """Require hard collision, FOV, wedge, and angle-range safety."""
        return (
            score_result[0][0] == 0
            and score_result[0][2] == 0
            and score_result[0][3] == 0
            and score_result[0][4] == 0
            and score_result[7].get("ok", True)
        )

    current = _score(
        target_positions,
        target_angles,
        guide_positions,
        guide_angles,
        constants,
        target_cegs=target_cegs,
        guide_cegs=guide_cegs,
        include_plugability=True,
    )
    diagnostics["initial_score"] = current[0]

    for round_number in range(max(1, int(max_rounds))):
        if resolved(current) and not current[8].get("misaligned", []) and not current[9].get("misaligned", []):
            diagnostics.update({"resolved": True, "final_score": current[0]})
            return target_angles, guide_angles, diagnostics
        current_target_fov = fov_status(current[3], constants.plate_radius)
        current_guide_fov = fov_status(current[4], constants.plate_radius)
        components = _joint_conflict_components(
            current[1], current[2],
            current_target_fov, current_guide_fov,
            current[8], current[9], current[7],
        )
        if not components:
            break
        diagnostics["rounds"] = round_number + 1
        improved = False
        for component_number, (target_indices, guide_indices) in enumerate(components):
            diagnostics["components"] += 1
            trial_targets, trial_guides, search_diag = joint_search(
                target_positions,
                target_angles,
                guide_positions,
                guide_angles,
                target_indices,
                guide_indices,
                constants,
                seed=seed + round_number * 1000 + component_number,
                deltas=(5.0, 10.0, 20.0, 45.0, 90.0),
                samples=max(80, (len(target_indices) + len(guide_indices)) * samples_per_probe),
                target_cegs=target_cegs,
                guide_cegs=guide_cegs,
                # Keep the stochastic inner loop focused on the expensive
                # collision/FOV objective.  Wedge and CEG-alignment checks
                # are still evaluated on the accepted candidate below and in
                # the final authoritative diagnostics.  Recomputing them for
                # every trial made the joint solver unexpectedly slow.
                include_plugability=False,
            )
            diagnostics["trials"] += int(search_diag.get("trials", 0))
            proposed = _score(
                target_positions,
                trial_targets,
                guide_positions,
                trial_guides,
                constants,
                target_cegs=target_cegs,
                guide_cegs=guide_cegs,
                include_plugability=True,
            )
            if proposed[0] < current[0]:
                target_angles = trial_targets
                guide_angles = trial_guides
                current = proposed
                improved = True
                if resolved(current):
                    break
        if not improved:
            break

    diagnostics.update({
        "resolved": resolved(current),
        "final_score": current[0],
    })
    return target_angles, guide_angles, diagnostics


def _configure_joint_workflow(
    tile_file: str | Path,
    inputs: Any,
    target_positions: pd.DataFrame,
    target_ids: list[str],
    target_angles: np.ndarray,
    target_cegs: np.ndarray,
    constants: HectorConstants,
    *,
    hexafile_out: str | Path | None,
    guidefile_out: str | Path | None,
    plot_file: str | Path | None,
    seed: int,
    joint_candidate_pool: int,
    joint_beam_width: int,
    joint_samples_per_probe: int,
    verbose: bool,
) -> ConfigurationResult:
    """Run the bounded all-class joint configuration workflow."""

    started = perf_counter()
    flags: list[str] = []
    standard_rows = inputs.fdata.loc[
        pd.to_numeric(inputs.fdata["type"], errors="coerce").eq(0)
    ].copy().reset_index(drop=True)
    cleaned_standards = _clean_standard_candidates(
        standard_rows, target_positions, target_angles, constants
    )
    standard_pool = _joint_candidate_pool(
        standard_rows,
        cleaned_standards,
        minimum=constants.nstdprobes,
        limit=joint_candidate_pool,
    )
    selected_standards, combined_positions, combined_angles, combined_cegs, standard_score = (
        _select_joint_standards(
            target_positions,
            target_angles,
            target_cegs,
            standard_pool,
            constants,
            candidate_pool=joint_candidate_pool,
        )
    )
    if len(selected_standards) < constants.nstdprobes:
        flags.append("StdFail")
    combined_ids = target_ids + selected_standards["ID"].astype(str).tolist()

    guide_rows = inputs.fdata.loc[
        pd.to_numeric(inputs.fdata["type"], errors="coerce").eq(2)
    ].copy().reset_index(drop=True)
    if len(guide_rows) < constants.ngprobesmax:
        flags.append("GuideFail")
    selected_guides, guide_angles, guide_cegs, guide_score = _select_joint_guides(
        combined_positions,
        combined_angles,
        guide_rows,
        constants,
        candidate_pool=joint_candidate_pool,
        beam_width=joint_beam_width,
    )
    if len(selected_guides) < constants.ngprobesmax:
        flags.append("GuideFail")

    combined_angles, guide_angles, joint_diag = _joint_optimize_configuration(
        combined_positions,
        combined_angles,
        combined_cegs,
        selected_guides[["x", "y"]].reset_index(drop=True),
        guide_angles,
        guide_cegs,
        constants,
        seed=seed,
        samples_per_probe=joint_samples_per_probe,
    )
    target_polygons = make_polygons(combined_positions, combined_angles, constants)
    guide_polygons = make_polygons(
        selected_guides[["x", "y"]].reset_index(drop=True), guide_angles, constants, guide=True
    )
    target_conflicts = find_probe_conflicts(target_polygons)
    guide_conflicts = find_guide_conflicts(target_polygons, guide_polygons)
    target_fov = fov_status(target_polygons, constants.plate_radius)
    guide_fov = fov_status(guide_polygons, constants.plate_radius)
    target_angle_status = angle_plugability_status(
        combined_positions, combined_angles, combined_cegs, constants
    )
    guide_angle_status = angle_plugability_status(
        selected_guides[["x", "y"]].reset_index(drop=True),
        guide_angles,
        guide_cegs,
        constants,
        guide=True,
    )
    wedges = wedge_status(
        combined_positions,
        combined_angles,
        selected_guides[["x", "y"]].reset_index(drop=True),
        guide_angles,
        constants,
    )
    if not target_conflicts.empty:
        flags.append("TargetFail")
    if not guide_conflicts.empty:
        flags.append("GuideFail")
    if not target_fov["ok"] or not guide_fov["ok"]:
        flags.append("FieldFail")
    if not target_angle_status["ok"] or not guide_angle_status["ok"] or not wedges["ok"]:
        flags.append("PlugFail")
    flags = list(dict.fromkeys(flags)) or ["none"]

    active_guide_ids = selected_guides["ID"].astype(str).tolist()
    config = {
        "pos": combined_positions.copy().set_index(pd.Index(combined_ids, name="ID")),
        "angs": np.asarray(combined_angles),
        "cegs": np.asarray(combined_cegs, dtype=int),
        "standard_indices": list(range(
            max(0, len(combined_positions) - constants.nstdprobes),
            len(combined_positions),
        )),
        "gcegs": np.asarray(guide_cegs, dtype=int),
        "gpos": selected_guides[["x", "y"]].copy().set_index(
            pd.Index(active_guide_ids, name="ID")
        ),
        "gangs": np.asarray(guide_angles),
        "flags": ",".join(flags),
        "guide_count": len(active_guide_ids),
        "manual_intervention": flags != ["none"],
        "target_conflicts": target_conflicts,
        "guide_conflicts": guide_conflicts,
        "target_fov": target_fov,
        "guide_fov": guide_fov,
        "target_angle_status": target_angle_status,
        "guide_angle_status": guide_angle_status,
        "wedge_status": wedges,
        "target_search": {"mode": "joint-initial", "score": standard_score},
        "joint_search": joint_diag,
        "joint_runtime_seconds": perf_counter() - started,
        "joint_standard_score": standard_score,
        "joint_guide_score": guide_score,
    }
    internal = type("InternalConfig", (), {
        "target_ids": combined_ids,
        "target_positions": combined_positions,
        "target_angles": combined_angles,
        "guide_ids": active_guide_ids,
        "guide_positions": selected_guides[["x", "y"]].reset_index(drop=True),
        "guide_angles": guide_angles,
    })()
    outputs = build_output_tables(inputs, internal, constants)
    if hexafile_out is not None:
        hector_write_tables(outputs, hexafile_out, guidefile_out)
    if plot_file is not None:
        plot_configured_field(
            combined_positions,
            combined_angles,
            selected_guides[["x", "y"]].reset_index(drop=True),
            guide_angles,
            constants,
            plot_file,
            flags=flags,
            standard_indices=range(
                max(0, len(combined_positions) - constants.nstdprobes),
                len(combined_positions),
            ),
        )
    if verbose:
        print(
            f"Joint candidate pool selected {len(selected_standards)} standards and "
            f"{len(active_guide_ids)} guides; score {joint_diag.get('initial_score')} -> "
            f"{joint_diag.get('final_score')} in {config['joint_runtime_seconds']:.2f} s."
        )
        print(f"Finished with flags: {config['flags']}")
    return ConfigurationResult(config, outputs["hexas"], outputs["guides"], inputs, constants)


def configure_hector(
    tile_file: str | Path,
    guide_file: str | Path,
    *,
    hexafile_out: str | Path | None = None,
    guidefile_out: str | Path | None = None,
    plot_file: str | Path | None = None,
    constants_path: str | Path | None = None,
    constants: HectorConstants | None = None,
    visualise: bool = False,
    seed: int = 2024,
    extended_samples_per_probe: int = 400,
    search_strategy: str = "joint",
    joint_candidate_pool: int = 12,
    joint_beam_width: int = 4,
    joint_samples_per_probe: int = 80,
    verbose: bool = True,
) -> ConfigurationResult:
    """Configure one HECTOR field.

    ``visualise`` is accepted for R-call compatibility.  The Python solver
    does not pause between stochastic trials; use ``plot_file`` for a stable
    diagnostic plot and the Shiny app for live/manual inspection.  If the
    automatic search cannot place all six guides, this function still writes
    the best available six-candidate output and sets ``GuideFail`` so the app
    can be used for manual repair.

    ``extended_samples_per_probe`` controls the final bounded stochastic
    search for hard target-guide clashes. The default of 400 gives the Python
    port a little more opportunity than the R-sized fallback to find a narrow
    valid angular interval; increasing it trades runtime for a greater chance
    of finding a solution.

    ``search_strategy="joint"`` is the default Python workflow. It selects a
    bounded pool of standards and guides up front, then optimises targets,
    standards, and guides through shared conflict components. This can avoid
    the staged solver freezing a target angle that later blocks a guide.
    ``joint_candidate_pool``, ``joint_beam_width``, and
    ``joint_samples_per_probe`` control its bounded runtime/quality
    trade-off. The joint Python solver is not validated to reproduce the R
    package: in testing, the Python implementation performed substantially
    worse than the parent R `hectorconfig` 0.1.18 solver. Compare outputs
    against R before using a configuration for observing. Set
    ``search_strategy="staged"`` explicitly to use the legacy R-like Python
    workflow.
    """

    if (hexafile_out is None) != (guidefile_out is None):
        raise ValueError("hexafile_out and guidefile_out must be supplied together.")
    if search_strategy not in {"staged", "joint"}:
        raise ValueError("search_strategy must be 'staged' or 'joint'.")
    if joint_candidate_pool < 1 or joint_beam_width < 1 or joint_samples_per_probe < 1:
        raise ValueError("joint search limits must all be at least 1.")
    constants = constants or load_constants(constants_path)
    tables = hector_read_tables(tile_file, guide_file)
    inputs = prepare_field(tables["tile"], tables["guide"], constants)

    # The R workflow takes the first required number of type-1 targets after
    # filtering.  Keep that deterministic rule here so R/Python comparisons
    # start with the same target list.
    tile_rows = inputs.fdata.loc[
        pd.to_numeric(inputs.fdata["type"], errors="coerce").eq(1)
    ].copy()
    target_rows = tile_rows.iloc[: constants.dngalprobes].reset_index(drop=True)
    target_positions = _positions_from_rows(target_rows)
    target_ids = _ids_from_rows(target_rows)
    target_cegs = choose_cegs(target_positions, constants)
    target_angles = first_guess_angles(target_positions, target_cegs, constants)

    if search_strategy == "joint":
        # Deliberately enter the joint experiment before the target-only
        # optimiser. The point of this mode is to let the later standards
        # and guides influence the initial target rotations as well.
        if verbose:
            print(f"Configuring field {Path(tile_file).resolve()}")
            print("Using joint target/standard/guide search (Python; compare with R before observing)")
        return _configure_joint_workflow(
            tile_file,
            inputs,
            target_positions,
            target_ids,
            target_angles,
            target_cegs,
            constants,
            hexafile_out=hexafile_out,
            guidefile_out=guidefile_out,
            plot_file=plot_file,
            seed=seed,
            joint_candidate_pool=joint_candidate_pool,
            joint_beam_width=joint_beam_width,
            joint_samples_per_probe=joint_samples_per_probe,
            verbose=verbose,
        )

    target_angles, target_diag = optimize_targets(
        target_positions, target_angles, constants, cegs=target_cegs, seed=seed
    )
    target_cegs = np.asarray(target_diag.get("cegs", target_cegs), dtype=int)

    if verbose:
        print(f"Configuring field {Path(tile_file).resolve()}")
        print(f"Initial/optimised target conflicts: {target_diag.get('initial_conflicts', 0)} -> "
              f"{len(find_probe_conflicts(make_polygons(target_positions, target_angles, constants)))}")

    # Set failure flags from the final authoritative configuration below.
    # An initial target clash may be resolved during the standard-target pass,
    # so recording ``TargetFail`` here would leave stale diagnostics behind.
    flags: list[str] = []

    # Match the R engine's positional cleaning before adding standards: choose
    # candidates outside the nearest target polygon, remove candidates close
    # to the plate edge, and try them from the field centre outward.
    standard_rows = inputs.fdata.loc[
        pd.to_numeric(inputs.fdata["type"], errors="coerce").eq(0)
    ].copy().reset_index(drop=True)
    standard_rows = _clean_standard_candidates(
        standard_rows, target_positions, target_angles, constants
    )
    standard_added = 0
    failed_standards: list[tuple[str, float, float, float, int]] = []
    for row in standard_rows.itertuples(index=False):
        if standard_added >= constants.nstdprobes:
            break
        candidate = pd.DataFrame({"x": [float(row.x)], "y": [float(row.y)]})
        trial_positions = pd.concat([target_positions, candidate], ignore_index=True)
        trial_ids = target_ids + [str(row.ID)]
        # Existing exit choices and angles are held fixed. The R engine only
        # selects a new exit/angle for the standard being added.
        new_ceg = int(choose_cegs(trial_positions, constants)[-1])
        trial_cegs = np.append(target_cegs, new_ceg)
        trial_angles = np.append(
            target_angles,
            first_guess_angles(candidate, [new_ceg], constants)[0],
        )
        trial_angles, trial_diag = optimize_targets(
            trial_positions,
            trial_angles,
            constants,
            cegs=trial_cegs,
            seed=seed + standard_added + 1,
        )
        conflicts = find_probe_conflicts(make_polygons(trial_positions, trial_angles, constants))
        fov = fov_status(make_polygons(trial_positions, trial_angles, constants), constants.plate_radius)
        # Accept a standard only when the complete trial is valid. A failed
        # candidate is held aside and may be retained later for manual repair,
        # while subsequent standard candidates still get a chance.
        if conflicts.empty and fov["ok"]:
            target_positions, target_ids, target_angles = trial_positions, trial_ids, trial_angles
            target_cegs = np.asarray(trial_diag.get("cegs", trial_cegs), dtype=int)
            standard_added += 1
            message = "configured"
        else:
            failed_cegs = np.asarray(trial_diag.get("cegs", trial_cegs), dtype=int)
            failed_standards.append(
                (str(row.ID), float(row.x), float(row.y), float(trial_angles[-1]), int(failed_cegs[-1]))
            )
            message = "retained for manual intervention"
        if verbose:
            print(f"Standard {standard_added + (0 if message == 'configured' else 1)}: {message}")
    # If a field has fewer than two automatically usable standards, retain
    # failed candidates so the Python checker/app can still be used. This is a
    # deliberate manual-intervention extension of the R stop condition.
    if standard_added < constants.nstdprobes:
        flags.append("StdFail")
        for standard_id, x, y, angle, ceg in failed_standards:
            if standard_added >= constants.nstdprobes:
                break
            target_positions = pd.concat(
                [target_positions, pd.DataFrame({"x": [x], "y": [y]})], ignore_index=True
            )
            target_ids.append(standard_id)
            target_angles = np.append(target_angles, angle)
            target_cegs = np.append(target_cegs, ceg)
            standard_added += 1
    # Guide candidates are first ordered with the same distance and quadrant
    # rules as R. Candidates removed by that positional pass are appended to
    # the queue afterwards, so difficult fields still get every candidate.
    guide_rows = inputs.fdata.loc[
        pd.to_numeric(inputs.fdata["type"], errors="coerce").eq(2)
    ].copy().reset_index(drop=True)
    if len(guide_rows) < constants.ngprobesmax:
        raise ValueError(
            f"The input contains only {len(guide_rows)} guide candidates; "
            f"{constants.ngprobesmax} are required."
        )
    cleaned_guides = _clean_guide_candidates(
        guide_rows, target_positions, target_angles, constants
    )
    cleaned_ids = set(cleaned_guides["ID"].astype(str))
    remaining_guides = guide_rows.loc[~guide_rows["ID"].astype(str).isin(cleaned_ids)]
    guide_rows = pd.concat([cleaned_guides, remaining_guides], ignore_index=True)

    active_guides = pd.DataFrame(columns=["x", "y"], dtype=float)
    active_ids: list[str] = []
    active_angles = np.asarray([], dtype=float)
    failed: list[tuple[str, float, float, float]] = []
    guide_attempts: list[dict[str, Any]] = []

    for candidate_number, row in enumerate(guide_rows.itertuples(index=False)):
        if len(active_guides) >= constants.ngprobesmax:
            break
        candidate = pd.DataFrame({"x": [float(row.x)], "y": [float(row.y)]})
        trial_guides = pd.concat([active_guides, candidate], ignore_index=True)
        trial_cegs = choose_cegs(trial_guides, constants)
        trial_angles = first_guess_angles(trial_guides, trial_cegs, constants, guide=True)
        trial_angles[:-1] = active_angles
        # First check the natural angle.  If it clashes, the joint search below
        # is allowed to move the conflicted target(s) as well as the guide.
        current = _guide_trial(
            target_positions, target_angles, trial_guides, trial_angles, constants
        )
        _, trial_target_angles, trial_guide_angles, target_conflicts, guide_conflicts, target_fov, guide_fov = current
        target_indices: set[int] = set()
        guide_indices: set[int] = {len(trial_guides) - 1}
        if not guide_conflicts.empty:
            for conflict in guide_conflicts.itertuples(index=False):
                if conflict.ConflictType == "target-guide" and int(conflict.Probe_2nd) == len(trial_guides):
                    target_indices.add(int(conflict.Probe_1st) - 1)
                elif conflict.ConflictType == "guide-guide":
                    guide_indices.add(int(conflict.Probe_1st) - 1)
                    guide_indices.add(int(conflict.Probe_2nd) - 1)

        search_diag: dict[str, Any] = {"resolved": target_conflicts.empty and guide_conflicts.empty and target_fov["ok"] and guide_fov["ok"], "trials": 0}
        if not search_diag["resolved"]:
            candidate_targets, candidate_guides, _, search_diag = minimize_guide_conflicts(
                target_positions,
                target_angles,
                target_cegs,
                trial_guides,
                trial_angles,
                trial_cegs,
                constants,
                guide_conflicts,
                seed=seed + 1000 + candidate_number,
                extended_samples_per_probe=extended_samples_per_probe,
            )
            check = _guide_trial(
                target_positions, candidate_targets, trial_guides, candidate_guides, constants
            )
            _, checked_targets, checked_guides, target_conflicts, guide_conflicts, target_fov, guide_fov = check
            if search_diag.get("resolved") and target_conflicts.empty and guide_conflicts.empty and target_fov["ok"] and guide_fov["ok"]:
                target_angles = checked_targets
                trial_guide_angles = checked_guides
                search_diag["accepted"] = True
            else:
                search_diag["accepted"] = False
        else:
            search_diag["accepted"] = True

        accepted = bool(search_diag.get("accepted"))
        guide_attempts.append({
            "candidate": candidate_number + 1,
            "ID": str(row.ID),
            "accepted": accepted,
            "search": search_diag,
            "remaining_conflicts": len(guide_conflicts),
        })
        if accepted:
            active_guides = trial_guides
            active_ids.append(str(row.ID))
            active_angles = np.asarray(trial_guide_angles, dtype=float)
            if verbose:
                print(f"Guide {len(active_guides)} configured: {row.ID}")
        else:
            failed.append((str(row.ID), float(row.x), float(row.y), float(trial_guide_angles[-1])))
            if verbose:
                print(f"Guide candidate retained for manual intervention: {row.ID}")

    # Fill missing slots only after all candidates were tried.  This preserves
    # exactly six rows for the manual checker even when the last few candidates
    # still have unresolved overlaps.
    for guide_id, x, y, angle in failed:
        if len(active_guides) >= constants.ngprobesmax:
            break
        active_guides = pd.concat([active_guides, pd.DataFrame({"x": [x], "y": [y]})], ignore_index=True)
        active_ids.append(guide_id)
        active_angles = np.append(active_angles, angle)

    # Rebuild everything from the accepted arrays for the final, authoritative
    # diagnostics.  No candidate is considered successful merely because its
    # centre is inside the plate.
    target_polygons = make_polygons(target_positions, target_angles, constants)
    guide_polygons = make_polygons(active_guides, active_angles, constants, guide=True)
    target_conflicts = find_probe_conflicts(target_polygons)
    guide_conflicts = find_guide_conflicts(target_polygons, guide_polygons)
    target_fov = fov_status(target_polygons, constants.plate_radius)
    guide_fov = fov_status(guide_polygons, constants.plate_radius)
    guide_cegs = choose_cegs(active_guides, constants)
    target_angle_status = angle_plugability_status(
        target_positions, target_angles, target_cegs, constants
    )
    guide_angle_status = angle_plugability_status(
        active_guides, active_angles, guide_cegs, constants, guide=True
    )
    wedges = wedge_status(
        target_positions, target_angles, active_guides, active_angles, constants
    )
    if not target_conflicts.empty:
        flags.append("TargetFail")
    if len(active_guides) < constants.ngprobesmax or not guide_conflicts.empty:
        flags.append("GuideFail")
    if not target_fov["ok"] or not guide_fov["ok"]:
        flags.append("FieldFail")
    if not target_angle_status["ok"] or not guide_angle_status["ok"] or not wedges["ok"]:
        flags.append("PlugFail")
    flags = list(dict.fromkeys(flags))
    if not flags:
        flags = ["none"]
    manual_intervention = flags != ["none"]

    config = {
        "pos": target_positions.copy().set_index(pd.Index(target_ids, name="ID")),
        "angs": np.asarray(target_angles),
        "cegs": np.asarray(target_cegs, dtype=int),
        "standard_indices": list(range(
            max(0, len(target_positions) - constants.nstdprobes),
            len(target_positions),
        )),
        "gcegs": np.asarray(guide_cegs, dtype=int),
        "gpos": active_guides.copy().set_index(pd.Index(active_ids, name="ID")),
        "gangs": np.asarray(active_angles),
        "flags": ",".join(flags),
        "guide_count": len(active_guides),
        "manual_intervention": manual_intervention,
        "target_conflicts": target_conflicts,
        "guide_conflicts": guide_conflicts,
        "target_fov": target_fov,
        "guide_fov": guide_fov,
        "target_angle_status": target_angle_status,
        "guide_angle_status": guide_angle_status,
        "wedge_status": wedges,
        "guide_attempts": guide_attempts,
        "target_search": target_diag,
    }
    result_config = type("InternalConfig", (), {
        "target_ids": target_ids,
        "target_positions": target_positions,
        "target_angles": target_angles,
        "guide_ids": active_ids,
        "guide_positions": active_guides,
        "guide_angles": active_angles,
    })()
    outputs = build_output_tables(inputs, result_config, constants)
    if hexafile_out is not None:
        hector_write_tables(outputs, hexafile_out, guidefile_out)
    if plot_file is not None:
        plot_configured_field(
            target_positions, target_angles, active_guides, active_angles,
            constants,
            plot_file,
            flags=flags,
            standard_indices=range(
                max(0, len(target_positions) - constants.nstdprobes),
                len(target_positions),
            ),
        )
    if verbose:
        if flags == ["none"]:
            print(f"Configured {len(active_guides)} guide stars without automatic conflicts.")
        else:
            print(
                f"Retained {len(active_guides)} guide candidates for manual intervention; "
                f"automatic configuration is not valid. Flags: {','.join(flags)}"
            )
    return ConfigurationResult(config, outputs["hexas"], outputs["guides"], inputs, constants)
