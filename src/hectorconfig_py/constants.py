"""Constants used by the HECTOR configuration engine.

The source YAML is kept in the package data directory.  Keeping the derived
values here, rather than scattering numbers through the solver, is the main
protection against the hard-coded-path and stale-constant problems that the R
package had early in its development.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class HectorConstants:
    """Static and derived dimensions used by the HECTOR geometry.

    The R engine created many of these values as global variables while it
    loaded.  Here they are kept together in one immutable object, which makes
    it explicit which constants a function is using and makes test overrides
    safe.  All distances are millimetres; angles are radians unless a name or
    comment explicitly says degrees.
    """
    circular_magnet_radius: float
    rectangle_magnet_width: float
    rectangle_magnet_length: float
    circular_rectangle_magnet_distance: float
    plate_radius: float
    skybuffer: float = 8.0
    wceg: float = 30.0
    fbr: float = 45.0
    tip_w: float = 14.5 / np.sqrt(2.0)
    tip_l: float = 14.5 / np.sqrt(2.0)
    probe_w: float = 18.5
    probe_l: float = 45.0
    cable_w: float = 8.5
    cable_l: float = 45.0
    dngalprobes: int = 19
    nstdprobes: int = 2
    ngprobesmin: int = 3
    ngprobesmax: int = 6
    gtip_w: float = 14.5 / np.sqrt(2.0)
    gtip_l: float = 14.5 / np.sqrt(2.0)
    gprobe_w: float = 14.5
    gprobe_l: float = 45.0
    gcable_w: float = 8.5
    gcable_l: float = 45.0
    delta_poly: float = 5.0

    @property
    def fov(self) -> float:
        """Diameter of the circular HECTOR field of view in millimetres."""
        return 2.0 * self.plate_radius

    @property
    def excl_radius(self) -> float:
        """Conservative circular exclusion radius for a science probe head."""
        return np.sqrt(2.0 * (self.tip_w / 2.0) ** 2) * 1.05

    @property
    def gexcl_radius(self) -> float:
        """Circular exclusion radius for a guide probe head."""
        return np.sqrt(2.0 * (self.gtip_w / 2.0) ** 2)

    @property
    def ceg_positions(self) -> np.ndarray:
        """Three cable-exit positions as columns ``x, y, angle_degrees``."""
        degrees = np.array([-90.0, 30.0, 150.0])
        radians = np.deg2rad(degrees)
        return np.column_stack(
            [self.plate_radius * np.cos(radians),
             self.plate_radius * np.sin(radians),
             degrees]
        )

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.update({
            "fov": self.fov,
            "excl_radius": self.excl_radius,
            "gexcl_radius": self.gexcl_radius,
            "ceg_positions": self.ceg_positions,
        })
        return values


def _package_constants_path() -> Path:
    return Path(resources.files("hectorconfig_py").joinpath("data/HECTOR_CONSTANTS.yaml"))


def _read_yaml(path: str | Path | None) -> Mapping[str, Any]:
    if path is None:
        path = _package_constants_path()
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return values


def load_constants(path: str | Path | None = None, **overrides: Any) -> HectorConstants:
    """Load the shared HECTOR YAML constants and return derived parameters.

    ``overrides`` is useful for controlled experiments, for example
    ``load_constants(plate_radius=226.0)``.  The YAML uses the original R
    name ``HECTOR_plate_radius``; the Python object uses the shorter
    ``plate_radius`` name after loading.
    """

    raw = dict(_read_yaml(path))
    values: dict[str, Any] = {
        "circular_magnet_radius": raw.get("circular_magnet_radius", 6.2),
        "rectangle_magnet_width": raw.get("rectangle_magnet_width", 12.4),
        "rectangle_magnet_length": raw.get("rectangle_magnet_length", 22.0),
        "circular_rectangle_magnet_distance": raw.get(
            "circular_rectangle_magnet_distance", 10.0
        ),
        "plate_radius": raw.get("HECTOR_plate_radius", 226.0),
    }
    values.update(overrides)
    return HectorConstants(**values)


def hector_engine_parameters(
    constants_path: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return the engine parameters as a plain dictionary.

    This mirrors the diagnostic use of ``hector_engine_parameters()`` in the
    R package and is handy when checking that Python loaded the intended
    constants file.
    """

    return load_constants(constants_path, **overrides).as_dict()
