"""Standalone validation of configured HECTOR CSV tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import HectorConstants, load_constants
from .engine import angle_plugability_status, infer_cegs_from_angles
from .geometry import find_guide_conflicts, find_probe_conflicts, fov_status, make_polygons, wedge_status
from .io import hector_read_tables


@dataclass
class CheckResult:
    """All diagnostics produced by :func:`check_configuration`."""

    flags: list[str]
    target_conflicts: pd.DataFrame
    guide_conflicts: pd.DataFrame
    target_fov: dict[str, object]
    guide_fov: dict[str, object]
    plugability: dict[str, object]
    target_count: int
    standard_count: int
    guide_count: int

    @property
    def ok(self) -> bool:
        """Whether the configured tables pass all automatic checks."""

        return self.flags == ["none"]

    def summary_lines(self) -> list[str]:
        """Return short lines suitable for a terminal or Shiny status panel."""

        lines = [
            f"Flags: {','.join(self.flags)}",
            f"Targets: {self.target_count}",
            f"Standards: {self.standard_count}",
            f"Guides: {self.guide_count}",
            f"Target/standard conflicts: {len(self.target_conflicts)}",
            f"Guide conflicts: {len(self.guide_conflicts)}",
            f"Targets outside field: {self.target_fov['outside']}",
            f"Guides outside field: {self.guide_fov['outside']}",
            f"Minimum target clearance (mm): {self.target_fov['minimum_clearance_mm']:.6g}",
            f"Minimum guide clearance (mm): {self.guide_fov['minimum_clearance_mm']:.6g}",
            f"Plugability: {self.plugability['ok']}",
        ]
        return lines

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def check_configuration(
    tile_file: str | Path,
    guide_file: str | Path,
    *,
    constants_path: str | Path | None = None,
    constants: HectorConstants | None = None,
) -> CheckResult:
    """Validate already configured tile and guide CSVs.

    This function does not run the automatic solver and does not change either
    input file.  It is therefore the quickest way to inspect the output from
    either the R package or the Python package before opening the manual app.
    """

    constants = constants or load_constants(constants_path)
    tables = hector_read_tables(tile_file, guide_file)
    tile = tables["tile"]
    guide = tables["guide"]
    required_tile = {"probe", "x", "y", "angs"}
    required_guide = {"x", "y", "angs"}
    missing_tile = required_tile.difference(tile.columns)
    missing_guide = required_guide.difference(guide.columns)
    if missing_tile:
        raise ValueError(
            "Tile table is not configured; missing column(s): "
            + ", ".join(sorted(missing_tile))
        )
    if missing_guide:
        raise ValueError(
            "Guide table is not configured; missing column(s): "
            + ", ".join(sorted(missing_guide))
        )

    probe_numbers = pd.to_numeric(tile["probe"], errors="coerce")
    configured = tile.loc[probe_numbers > 0].copy()
    if configured.empty:
        raise ValueError("The tile table contains no configured probes with probe > 0.")

    target_positions = configured[["x", "y"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    target_angles = pd.to_numeric(configured["angs"], errors="coerce").to_numpy(float)
    guide_positions = guide[["x", "y"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    guide_angles = pd.to_numeric(guide["angs"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(target_positions.to_numpy())) or not np.all(np.isfinite(target_angles)):
        raise ValueError("Configured tile positions or angles contain non-finite values.")
    if not np.all(np.isfinite(guide_positions.to_numpy())) or not np.all(np.isfinite(guide_angles)):
        raise ValueError("Configured guide positions or angles contain non-finite values.")

    target_polygons = make_polygons(target_positions, target_angles, constants)
    guide_polygons = make_polygons(guide_positions, guide_angles, constants, guide=True)
    target_conflicts = find_probe_conflicts(target_polygons)
    guide_conflicts = find_guide_conflicts(target_polygons, guide_polygons)
    target_fov = fov_status(target_polygons, constants.plate_radius)
    guide_fov = fov_status(guide_polygons, constants.plate_radius)
    target_cegs = infer_cegs_from_angles(target_positions, target_angles, constants)
    guide_cegs = infer_cegs_from_angles(
        guide_positions, guide_angles, constants, guide=True
    )
    target_angle_status = angle_plugability_status(
        target_positions, target_angles, target_cegs, constants
    )
    guide_angle_status = angle_plugability_status(
        guide_positions, guide_angles, guide_cegs, constants, guide=True
    )
    wedges = wedge_status(
        target_positions, target_angles, guide_positions, guide_angles, constants
    )
    plugability = {
        "ok": target_angle_status["ok"] and guide_angle_status["ok"] and wedges["ok"],
        "target_angles": target_angle_status,
        "guide_angles": guide_angle_status,
        "wedges": wedges,
    }

    type_values = pd.to_numeric(configured["type"], errors="coerce") if "type" in configured else pd.Series(dtype=float)
    standard_count = int(type_values.eq(0).sum())
    target_count = int(type_values.eq(1).sum()) if len(type_values) else len(configured)
    flags: list[str] = []
    if len(target_conflicts):
        flags.append("TargetFail")
    if standard_count < constants.nstdprobes:
        flags.append("StdFail")
    if len(guide_conflicts) or len(guide_positions) < constants.ngprobesmax:
        flags.append("GuideFail")
    if not target_fov["ok"] or not guide_fov["ok"]:
        flags.append("FieldFail")
    if not plugability["ok"]:
        flags.append("PlugFail")
    if not flags:
        flags = ["none"]
    return CheckResult(
        flags=flags,
        target_conflicts=target_conflicts,
        guide_conflicts=guide_conflicts,
        target_fov=target_fov,
        guide_fov=guide_fov,
        plugability=plugability,
        target_count=target_count,
        standard_count=standard_count,
        guide_count=len(guide_positions),
    )
