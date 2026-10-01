"""CSV input preparation and output-table construction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import HectorConstants


REQUIRED_COLUMNS = ["ID", "MagnetX", "MagnetY", "type"]
DERIVED_COLUMNS = ["x", "y", "rads", "angs", "azAngs", "angs_azAng"]


@dataclass
class FieldInputs:
    """The original tables plus the filtered coordinate subsets used by search."""
    tile: pd.DataFrame
    guide: pd.DataFrame
    fdata: pd.DataFrame
    sky_data: pd.DataFrame
    target_positions: pd.DataFrame
    standard_positions: pd.DataFrame
    guide_positions: pd.DataFrame


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    """Read one HECTOR CSV and fail early if its structural columns are absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    data = pd.read_csv(path, dtype={"ID": "string"})
    # The legacy R CSVs begin with an empty header for the old row-name column.
    # R reads this as ``X`` (after ``check.names=TRUE``), whereas pandas calls
    # it ``Unnamed: 0``. Preserve it under the R-compatible name: it is part
    # of the legacy table layout, and dropping it changes the output format
    # when a Python-written file is handed back to the R checker.
    if len(data.columns) and str(data.columns[0]).startswith("Unnamed:"):
        data = data.rename(columns={data.columns[0]: "X"})
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {', '.join(missing)}")
    return data


def hector_read_tables(tile_file: str | Path, guide_file: str | Path) -> dict[str, pd.DataFrame]:
    """Read a tile CSV and a guide-star CSV, preserving their source paths."""

    tile = _read_csv(tile_file, "Tile")
    guide = _read_csv(guide_file, "Guide")
    tile.attrs["source_path"] = str(Path(tile_file).resolve())
    guide.attrs["source_path"] = str(Path(guide_file).resolve())
    return {"tile": tile, "guide": guide}


def _numeric_series(data: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    """Convert a possibly absent CSV column to numeric values."""
    if column not in data:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def prepare_field(tile: pd.DataFrame, guide: pd.DataFrame, constants: HectorConstants) -> FieldInputs:
    """Prepare the input tables in the same units and categories as the R engine.

    Magnet coordinates arrive in microns and are converted to millimetres.
    Sky fibres are retained separately for output, while the solver receives
    science, standard, and guide candidates inside the conservative head
    exclusion radius.
    """

    tile = tile.copy()
    guide = guide.copy()
    tile["ID"] = tile["ID"].astype(str)
    guide["ID"] = guide["ID"].astype(str)
    # Sky fibres are not configurable probes, but the output should keep them.
    sky_mask = tile["ID"].str.contains("Sky", case=False, na=False)
    sky_data = tile.loc[sky_mask].copy()
    science = tile.loc[~sky_mask].copy()
    fdata = pd.concat([science, guide], ignore_index=True, sort=False)
    fdata["x"] = pd.to_numeric(fdata["MagnetX"], errors="coerce") / 1000.0
    fdata["y"] = pd.to_numeric(fdata["MagnetY"], errors="coerce") / 1000.0
    fdata["r"] = np.hypot(fdata["x"], fdata["y"])
    # Filter on probe-head clearance, not just centre position.  The later
    # polygon FOV check is stricter and catches a cable or ferrule outside the
    # plate after an angle has been chosen.
    keep = fdata["r"] < (constants.fov / 2.0 - constants.excl_radius)
    fdata = fdata.loc[keep.fillna(False)].copy()
    if len(fdata) < constants.dngalprobes:
        raise ValueError(
            f"The field contains fewer than {constants.dngalprobes} usable targets."
        )
    if fdata["ID"].duplicated().any():
        duplicate_ids = fdata.loc[fdata["ID"].duplicated(), "ID"].tolist()
        raise ValueError(f"Input IDs must be unique across target and guide tables: {duplicate_ids}")

    # Type 1 = galaxy target, 0 = standard star, 2 = guide star.
    type_values = pd.to_numeric(fdata["type"], errors="coerce")
    target_positions = fdata.loc[type_values.eq(1), ["x", "y"]].reset_index(drop=True)
    standard_positions = fdata.loc[type_values.eq(0), ["x", "y"]].reset_index(drop=True)
    guide_positions = fdata.loc[type_values.eq(2), ["x", "y"]].reset_index(drop=True)
    if len(target_positions) < constants.dngalprobes:
        raise ValueError(
            f"The field contains fewer than {constants.dngalprobes} type-1 galaxy targets."
        )
    return FieldInputs(
        tile=tile,
        guide=guide,
        fdata=fdata,
        sky_data=sky_data,
        target_positions=target_positions,
        standard_positions=standard_positions,
        guide_positions=guide_positions,
    )


def _azimuth_angles(positions: pd.DataFrame) -> np.ndarray:
    """Compute the R engine's azimuth reference angle in radians.

    Despite the historical column name ``azAngs``, the R engine stores this
    value in radians. It is the direction from a probe position toward the
    nearest field edge, with the same wrap convention used by the robot
    output.
    """
    angles = np.pi + np.arctan2(
        positions["y"].to_numpy(float), positions["x"].to_numpy(float)
    )
    angles = angles.copy()
    angles[angles > 1.5 * np.pi] -= 2.0 * np.pi
    angles[angles < -0.5 * np.pi] += 2.0 * np.pi
    return angles


def _relative_angles(angles: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Compute the wrapped probe angle relative to the field-edge azimuth."""
    relative = np.asarray(angles, dtype=float) - np.asarray(azimuth, dtype=float)
    relative = relative.copy()
    relative[relative > np.pi] -= 2.0 * np.pi
    relative[relative < 0.0] += 2.0 * np.pi
    return relative


def build_output_tables(
    inputs: FieldInputs,
    config,
    constants: HectorConstants,
) -> dict[str, pd.DataFrame]:
    """Attach final positions/angles to the original tile and guide metadata.

    Metadata columns come from the user's input; geometry columns are always
    rebuilt from the final configuration so a second pass cannot accidentally
    preserve stale angles or positions from an earlier output file.
    """

    target_ids = list(config.target_ids)
    tile_by_id = inputs.tile.copy()
    tile_by_id["ID"] = tile_by_id["ID"].astype(str)
    selected = tile_by_id.set_index("ID").reindex(target_ids).reset_index()
    if selected["ID"].isna().any():
        raise ValueError("Some configured target IDs could not be found in the tile table.")

    # A previously configured CSV may already contain ``probe``.  Drop all
    # generated columns before inserting the authoritative final values.
    # Match the R writer's canonical tile layout: generated geometry follows
    # ``probe, ID`` and precedes the non-derived input metadata. Dropping an
    # old ``probe`` column is intentional; otherwise a second save/reload
    # cycle can create duplicate probe columns that confuse legacy readers.
    tile_input_columns = [
        column for column in selected.columns
        if column not in {"ID", "probe", *DERIVED_COLUMNS}
    ]
    target_table = pd.DataFrame({
        "probe": np.arange(1, len(selected) + 1),
        "ID": target_ids,
        "x": config.target_positions["x"].to_numpy(),
        "y": config.target_positions["y"].to_numpy(),
        "rads": np.hypot(config.target_positions["x"], config.target_positions["y"]),
        "angs": np.asarray(config.target_angles),
    })
    target_azimuth = _azimuth_angles(config.target_positions)
    target_table["azAngs"] = target_azimuth
    target_table["angs_azAng"] = _relative_angles(
        np.asarray(config.target_angles), target_azimuth
    )
    target_table = pd.concat(
        [target_table.reset_index(drop=True), selected[tile_input_columns].reset_index(drop=True)],
        axis=1,
    )

    if len(inputs.sky_data):
        sky = inputs.sky_data.copy()
        sky["ID"] = sky["ID"].astype(str)
        sky_input_columns = [
            column for column in sky.columns
            if column not in {"ID", "probe", *DERIVED_COLUMNS}
        ]
        sky_table = pd.DataFrame({
            "probe": -99,
            "ID": sky["ID"].to_numpy(),
            "x": pd.to_numeric(sky["MagnetX"], errors="coerce").to_numpy() / 1000.0,
            "y": pd.to_numeric(sky["MagnetY"], errors="coerce").to_numpy() / 1000.0,
            "rads": -99.0,
            "angs": -99.0,
            "azAngs": -99.0,
            "angs_azAng": -99.0,
        })
        sky_table = pd.concat(
            [sky_table.reset_index(drop=True), sky[sky_input_columns].reset_index(drop=True)],
            axis=1,
        )
        hexas = pd.concat([target_table, sky_table], ignore_index=True, sort=False)
    else:
        hexas = target_table

    guide_by_id = inputs.guide.copy()
    guide_by_id["ID"] = guide_by_id["ID"].astype(str)
    guide_table = guide_by_id.set_index("ID").reindex(list(config.guide_ids)).reset_index()
    if guide_table["ID"].isna().any():
        raise ValueError("Some configured guide IDs could not be found in the guide table.")
    guide_input_columns = [
        column for column in guide_table.columns
        if column not in {"probe", *DERIVED_COLUMNS}
    ]
    guide_table = guide_table[guide_input_columns].reset_index(drop=True)
    guide_table["x"] = config.guide_positions["x"].to_numpy()
    guide_table["y"] = config.guide_positions["y"].to_numpy()
    guide_table["rads"] = np.hypot(config.guide_positions["x"], config.guide_positions["y"])
    guide_table["angs"] = np.asarray(config.guide_angles)
    guide_azimuth = _azimuth_angles(config.guide_positions)
    guide_table["azAngs"] = guide_azimuth
    guide_table["angs_azAng"] = _relative_angles(
        np.asarray(config.guide_angles), guide_azimuth
    )
    return {"hexas": hexas, "guides": guide_table}


def hector_write_tables(tables: dict[str, pd.DataFrame], hexafile_out: str | Path, guidefile_out: str | Path) -> None:
    """Write configured tables without adding a pandas index column."""

    if "hexas" not in tables or "guides" not in tables:
        raise ValueError("tables must contain 'hexas' and 'guides' data frames.")
    Path(hexafile_out).parent.mkdir(parents=True, exist_ok=True)
    Path(guidefile_out).parent.mkdir(parents=True, exist_ok=True)
    # Match the R writer's stable interchange conventions: no row index, no
    # automatic quoting of ordinary character fields, and explicit ``NA`` for
    # missing metadata rather than pandas' empty-field default. This matters
    # when a file is saved by the Python app and then reloaded by the R app.
    write_options = {
        "index": False,
        "na_rep": "NA",
        "quoting": csv.QUOTE_NONE,
        "escapechar": "\\",
        "lineterminator": "\n",
    }
    tables["hexas"].to_csv(hexafile_out, **write_options)
    tables["guides"].to_csv(guidefile_out, **write_options)
