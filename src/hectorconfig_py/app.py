"""Shiny-for-Python manual checker for configured HECTOR fields.

Run from the package directory with::

    shiny run hectorconfig_py.app:app

The app intentionally keeps the manual controls small: select a probe on the
plot, enter an angle in degrees, apply it, inspect the conflict list, and
download the repaired tables.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

from .constants import load_constants
from .configure import ConfigurationResult, configure_hector, refresh_result_tables
from .geometry import (
    find_guide_conflicts,
    find_probe_conflicts,
    fov_status,
    make_polygons,
    wedge_status,
)
from .engine import angle_plugability_status, infer_cegs_from_angles
from .io import hector_read_tables, prepare_field
from .plotting import make_field_figure


def _angle_from_tail_click(center_x: float, center_y: float, click_x: float, click_y: float) -> float:
    """Convert a plot click into the R engine's stored probe angle.

    The click indicates where the probe tail/cable should point. The stored
    angle describes the opposite local +x direction, hence the added pi.
    """
    angle = math.pi + math.atan2(click_y - center_y, click_x - center_x)
    if angle > 1.5 * math.pi:
        angle -= 2.0 * math.pi
    if angle < -0.5 * math.pi:
        angle += 2.0 * math.pi
    return angle


def _describe_angle_issues(
    status: dict[str, object],
    ids: list[str],
    label: str,
) -> list[str]:
    """Turn angle diagnostics into useful manual-repair messages."""
    invalid = [int(index) for index in status.get("invalid", [])]
    ranges = list(status.get("ranges_degrees", []))
    messages = []
    for index in invalid:
        probe_id = ids[index - 1] if 0 < index <= len(ids) else "unknown ID"
        allowed = ranges[index - 1] if 0 < index <= len(ranges) else (None, None)
        if allowed[0] is None:
            messages.append(f"{label} {index} ({probe_id}) is outside its allowed CEG angle range")
        else:
            messages.append(
                f"{label} {index} ({probe_id}) is outside its allowed CEG angle range "
                f"[{allowed[0]:.1f}, {allowed[1]:.1f}] degrees"
            )
    return messages


def _load_configured_tables(tile_path: str, guide_path: str) -> ConfigurationResult:
    """Load already configured CSVs for manual checking without re-running the solver.

    This is important for parity with the original R Shiny checker: the app
    must be able to open output from the R package as well as output from the
    Python automatic stage.
    """

    tables = hector_read_tables(tile_path, guide_path)
    constants = load_constants()
    inputs = prepare_field(tables["tile"], tables["guide"], constants)
    tile = tables["tile"].copy()
    guide = tables["guide"].copy()
    # Sky rows have probe = -99 and are not part of the editable configuration.
    probes = pd.to_numeric(tile["probe"], errors="coerce")
    selected = tile.loc[probes > 0].copy()
    if selected.empty or not {"x", "y", "angs"}.issubset(selected.columns):
        raise ValueError("The uploaded tile table does not look like a configured HECTOR output.")
    if not {"x", "y", "angs"}.issubset(guide.columns):
        raise ValueError("The uploaded guide table does not look like a configured HECTOR output.")
    target_positions = selected[["x", "y"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    target_ids = selected["ID"].astype(str).tolist()
    target_angles = pd.to_numeric(selected["angs"], errors="coerce").to_numpy(float)
    guide_positions = guide[["x", "y"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    guide_ids = guide["ID"].astype(str).tolist()
    guide_angles = pd.to_numeric(guide["angs"], errors="coerce").to_numpy(float)
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
    flags = []
    if len(target_conflicts):
        flags.append("TargetFail")
    configured_types = pd.to_numeric(selected["type"], errors="coerce") if "type" in selected else pd.Series(dtype=float)
    standard_indices = np.flatnonzero(configured_types.eq(0).to_numpy()).tolist()
    if int(configured_types.eq(0).sum()) < constants.nstdprobes:
        flags.append("StdFail")
    if len(guide_conflicts) or len(guide_positions) < constants.ngprobesmax:
        flags.append("GuideFail")
    if not target_fov["ok"] or not guide_fov["ok"]:
        flags.append("FieldFail")
    if not target_angle_status["ok"] or not guide_angle_status["ok"] or not wedges["ok"]:
        flags.append("PlugFail")
    config = {
        "pos": target_positions.set_index(pd.Index(target_ids, name="ID")),
        "angs": target_angles,
        "cegs": target_cegs,
        "standard_indices": standard_indices,
        "gcegs": guide_cegs,
        "gpos": guide_positions.set_index(pd.Index(guide_ids, name="ID")),
        "gangs": guide_angles,
        "flags": ",".join(flags) if flags else "none",
        "guide_count": len(guide_positions),
        "manual_intervention": bool(flags),
        "target_conflicts": target_conflicts,
        "guide_conflicts": guide_conflicts,
        "target_fov": target_fov,
        "guide_fov": guide_fov,
        "target_angle_status": target_angle_status,
        "guide_angle_status": guide_angle_status,
        "wedge_status": wedges,
        "target_search": {"loaded_configured_tables": True},
    }
    return ConfigurationResult(config, tile, guide, inputs, constants)


app_ui = ui.page_fluid(
    ui.h2("HECTOR configuration checker"),
    ui.p("Upload both CSV files and click Configure field. Double-click a probe head to select it, then click in the desired tail direction; the degree compass and orange preview arrow show the proposed rotation. Fix Probe commits it."),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("tile_file", "Hexabundle/tile CSV", accept=[".csv"]),
            ui.input_file("guide_file", "Guide-star CSV", accept=[".csv"]),
            ui.input_action_button("configure", "Configure field"),
            ui.hr(),
            ui.output_text("selected_probe"),
            ui.input_numeric("angle_degrees", "New angle (degrees)", value=0),
            ui.input_action_button("fix_probe", "Fix Probe"),
            ui.input_action_button("reset_probe", "Reset selected probe"),
            ui.input_action_button("list_conflicts", "List conflicted probes"),
            ui.hr(),
            ui.output_code("status"),
            ui.download_button("download_hexas", "Download hexas CSV"),
            ui.download_button("download_guides", "Download guides CSV"),
            ui.download_button("download_plot", "Download configuration PDF"),
        ),
        ui.output_plot("field_plot", height="750px", click=True, dblclick=True),
    ),
)


def server(input, output, session):
    """Create the reactive state for one browser session."""
    constants = load_constants()
    # Do not load or configure anything at startup. The user explicitly
    # selects both CSVs and starts the workflow with Configure field.
    state = reactive.Value(None)
    selected = reactive.Value(None)
    # Preview angles are separate from committed result angles. A click can
    # explore a direction repeatedly; Fix Probe explicitly commits it.
    preview_angles = reactive.Value(None)
    message = reactive.Value("Choose both CSV files, then click Configure field.")
    show_conflicts = reactive.Value(False)
    running = reactive.Value(False)

    def _working_angles(result: ConfigurationResult) -> tuple[np.ndarray, np.ndarray]:
        """Return preview angles when present, otherwise committed angles."""
        preview = preview_angles.get()
        if preview is None:
            return np.asarray(result.config["angs"], dtype=float), np.asarray(result.config["gangs"], dtype=float)
        return preview["angs"], preview["gangs"]

    def _load_result(tile_path: str, guide_path: str):
        """Load raw inputs through the solver or configured tables directly."""
        if running.get():
            message.set("A configuration run is already in progress; please wait for it to finish.")
            return
        running.set(True)
        message.set("Working—please wait for the configuration search to finish.")
        try:
            uploaded_tile = hector_read_tables(tile_path, guide_path)["tile"]
            if uploaded_tile is not None and "probe" in uploaded_tile.columns:
                result = _load_configured_tables(tile_path, guide_path)
            else:
                result = configure_hector(tile_path, guide_path, visualise=False, verbose=False)
            state.set(result)
            selected.set(None)
            preview_angles.set(None)
            show_conflicts.set(False)
            message.set(
                f"Configured {result.config['guide_count']} guide(s). Flags: {result.config['flags']}"
            )
        except Exception as exc:  # give the beginner a useful message in the app
            message.set(f"Configuration failed: {type(exc).__name__}: {exc}")
        finally:
            running.set(False)

    @reactive.effect
    @reactive.event(input.configure)
    def _configure_uploaded():
        tile = input.tile_file()
        guide = input.guide_file()
        if not tile or not guide:
            message.set("Choose both CSV files first.")
            return
        _load_result(tile[0]["datapath"], guide[0]["datapath"])

    @reactive.effect
    @reactive.event(input.field_plot_dblclick)
    def _select_probe():
        """Select the probe nearest a plot double-click."""
        result = state.get()
        click = input.field_plot_dblclick()
        if result is None or not click:
            return
        x, y = float(click.get("x", np.nan)), float(click.get("y", np.nan))
        if not np.isfinite(x + y):
            return
        target_positions = result.config["pos"].reset_index(drop=True)
        guide_positions = result.config["gpos"].reset_index(drop=True)
        target_distances = np.hypot(target_positions.x - x, target_positions.y - y)
        guide_distances = np.hypot(guide_positions.x - x, guide_positions.y - y)
        if len(guide_distances) and guide_distances.min() < target_distances.min():
            selection = ("guide", int(guide_distances.argmin()))
        else:
            selection = ("target", int(target_distances.argmin()))
        selected.set(selection)
        current_targets, current_guides = _working_angles(result)
        current_angle = current_guides[selection[1]] if selection[0] == "guide" else current_targets[selection[1]]
        ui.update_numeric("angle_degrees", value=float(np.degrees(current_angle) % 360.0))
        message.set("Probe selected. Click around its head to preview the tail direction, then click Fix Probe.")

    @reactive.effect
    @reactive.event(input.field_plot_click)
    def _orient_probe():
        """Preview an angle from a single click in the desired tail direction."""
        result = state.get()
        selection = selected.get()
        click = input.field_plot_click()
        if result is None or selection is None:
            message.set("Double-click a probe head first, then click in the desired tail direction.")
            return
        if not click:
            return
        click_x, click_y = float(click.get("x", np.nan)), float(click.get("y", np.nan))
        if not np.isfinite(click_x + click_y):
            return
        kind, index = selection
        positions = result.config["gpos"] if kind == "guide" else result.config["pos"]
        center = positions.reset_index(drop=True).iloc[index]
        distance = math.hypot(click_x - float(center.x), click_y - float(center.y))
        if distance < 1.0:
            message.set("Click farther from the probe head to choose a direction.")
            return
        current_targets, current_guides = _working_angles(result)
        new_angle = _angle_from_tail_click(float(center.x), float(center.y), click_x, click_y)
        new_targets = current_targets.copy()
        new_guides = current_guides.copy()
        if kind == "guide":
            new_guides[index] = new_angle
        else:
            new_targets[index] = new_angle
        preview_angles.set({"angs": new_targets, "gangs": new_guides})
        ui.update_numeric("angle_degrees", value=float(np.degrees(new_angle) % 360.0))
        message.set(
            f"Previewing {kind} {index + 1} at {np.degrees(new_angle) % 360.0:.1f} degrees. "
            "Click Fix Probe to save this angle."
        )

    @reactive.effect
    @reactive.event(input.fix_probe)
    def _fix_probe():
        """Apply the sidebar angle to the selected probe and refresh outputs."""
        result = state.get()
        selection = selected.get()
        if result is None or selection is None:
            message.set("Double-click a probe first.")
            return
        kind, index = selection
        angle = math.radians(float(input.angle_degrees()))
        current_targets, current_guides = _working_angles(result)
        if kind == "guide":
            current_guides[index] = angle
            current_gcegs = np.asarray(
                result.config.get(
                    "gcegs",
                    infer_cegs_from_angles(
                        result.config["gpos"].reset_index(drop=True),
                        current_guides,
                        constants,
                        guide=True,
                    ),
                ),
                dtype=int,
            ).copy()
            current_gcegs[index] = infer_cegs_from_angles(
                result.config["gpos"].reset_index(drop=True).iloc[[index]],
                [angle],
                constants,
                guide=True,
            )[0]
            result.config["gcegs"] = current_gcegs
        else:
            current_targets[index] = angle
            current_cegs = np.asarray(
                result.config.get(
                    "cegs",
                    infer_cegs_from_angles(
                        result.config["pos"].reset_index(drop=True),
                        current_targets,
                        constants,
                    ),
                ),
                dtype=int,
            ).copy()
            current_cegs[index] = infer_cegs_from_angles(
                result.config["pos"].reset_index(drop=True).iloc[[index]],
                [angle],
                constants,
            )[0]
            result.config["cegs"] = current_cegs
        result.config["angs"] = current_targets
        result.config["gangs"] = current_guides
        refresh_result_tables(result)
        preview_angles.set(None)
        message.set("Angle updated. Click List conflicted probes to re-check the field.")
        state.set(result)

    @reactive.effect
    @reactive.event(input.reset_probe)
    def _reset_probe():
        """Restore the automatic configuration by rerunning the source inputs."""
        result = state.get()
        selection = selected.get()
        if result is None or selection is None:
            return
        # Resetting is intentionally conservative: rerun the automatic solver
        # from the original input tables, which restores every angle. There is
        # no bundled fallback because the app no longer loads an example.
        tile_path = str(result.inputs.tile.attrs.get("source_path", ""))
        guide_path = str(result.inputs.guide.attrs.get("source_path", ""))
        if not tile_path or not guide_path:
            message.set("The original input paths are unavailable; reload both CSV files to reset the field.")
            return
        _load_result(tile_path, guide_path)

    @reactive.effect
    @reactive.event(input.list_conflicts)
    def _list_conflicts():
        """Recompute and display overlap and complete-polygon FOV diagnostics."""
        result = state.get()
        if result is None:
            message.set("Load a field first.")
            return
        target_positions = result.config["pos"].reset_index(drop=True)
        guide_positions = result.config["gpos"].reset_index(drop=True)
        target_angles, guide_angles = _working_angles(result)
        target_polygons = make_polygons(target_positions, target_angles, constants)
        guide_polygons = make_polygons(guide_positions, guide_angles, constants, guide=True)
        target_conflicts = find_probe_conflicts(target_polygons)
        guide_conflicts = find_guide_conflicts(target_polygons, guide_polygons)
        target_fov = fov_status(target_polygons, constants.plate_radius)
        guide_fov = fov_status(guide_polygons, constants.plate_radius)
        target_cegs = result.config.get("cegs")
        if target_cegs is None:
            target_cegs = infer_cegs_from_angles(target_positions, target_angles, constants)
        guide_cegs = result.config.get("gcegs")
        if guide_cegs is None:
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
        parts = [
            f"target/standard conflicts: {len(target_conflicts)}",
            f"guide conflicts: {len(guide_conflicts)}",
            f"plugability: {target_angle_status['ok'] and guide_angle_status['ok'] and wedges['ok']}",
        ]
        if not target_fov["ok"] or not guide_fov["ok"]:
            parts.append(f"outside field: targets={target_fov['outside']}, guides={guide_fov['outside']}")
        if len(target_conflicts):
            parts.append(target_conflicts.to_string(index=False))
        if len(guide_conflicts):
            parts.append(guide_conflicts.to_string(index=False))
        if not target_angle_status["ok"]:
            parts.extend(_describe_angle_issues(
                target_angle_status,
                result.config["pos"].index.astype(str).tolist(),
                "Target/standard probe",
            ))
        if not guide_angle_status["ok"]:
            parts.extend(_describe_angle_issues(
                guide_angle_status,
                result.config["gpos"].index.astype(str).tolist(),
                "Guide probe",
            ))
        if not wedges["ok"]:
            parts.append(f"wedge checks: {wedges['violations']}")
        message.set("\n".join(parts))
        show_conflicts.set(True)

    @render.plot
    def field_plot():
        """Render the current session state; this is the live plot in the app."""
        result = state.get()
        selection = selected.get()
        target_positions = pd.DataFrame(result.config["pos"]).reset_index(drop=True) if result else pd.DataFrame(columns=["x", "y"])
        guide_positions = pd.DataFrame(result.config["gpos"]).reset_index(drop=True) if result else pd.DataFrame(columns=["x", "y"])
        target_angles, guide_angles = _working_angles(result) if result else ([], [])
        standard_indices = (
            result.config.get("standard_indices")
            if result is not None
            else None
        )
        selected_target = selection[1] if selection and selection[0] == "target" else None
        selected_guide = selection[1] if selection and selection[0] == "guide" else None
        compass_position = None
        compass_angle = None
        compass_radius = None
        if result is not None and selection is not None:
            kind, index = selection
            selected_positions = guide_positions if kind == "guide" else target_positions
            selected_angles = guide_angles if kind == "guide" else target_angles
            if len(selected_positions) > index:
                point = selected_positions.iloc[index]
                compass_position = (float(point.x), float(point.y))
                compass_angle = float(selected_angles[index])
                compass_radius = (
                    constants.gprobe_l + constants.gcable_l + constants.gtip_l / 2.0
                    if kind == "guide"
                    else constants.probe_l + constants.cable_l + constants.tip_l / 2.0
                )
        fig, _ = make_field_figure(
            target_positions,
            target_angles,
            guide_positions,
            guide_angles,
            constants,
            selected_target=selected_target,
            selected_guide=selected_guide,
            standard_indices=standard_indices,
            compass_position=compass_position,
            compass_angle=compass_angle,
            compass_radius=compass_radius,
            title="HECTOR configuration",
        )
        return fig

    @render.text
    def selected_probe():
        value = selected.get()
        return "Selected: none" if value is None else f"Selected: {value[0]} {value[1] + 1}"

    @render.code
    def status():
        return message.get()

    @render.download_button(filename="hector_hexas_configured.csv")
    def download_hexas():
        """Provide the current hexabundle table as a browser download."""
        result = state.get()
        if result is None:
            return None
        path = Path(tempfile.mkstemp(suffix=".csv")[1])
        result.hexas.to_csv(
            path,
            index=False,
            na_rep="NA",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            lineterminator="\n",
        )
        # Shiny's file-download renderer expects a path-like string for an
        # existing file. Returning a pathlib.Path is interpreted as an
        # iterable stream by some Shiny versions and raises ``Path object is
        # not iterable`` when the browser requests the download.
        return str(path)

    @render.download_button(filename="hector_guides_configured.csv")
    def download_guides():
        """Provide the current guide table as a browser download."""
        result = state.get()
        if result is None:
            return None
        path = Path(tempfile.mkstemp(suffix=".csv")[1])
        result.guides.to_csv(
            path,
            index=False,
            na_rep="NA",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            lineterminator="\n",
        )
        return str(path)

    @render.download_button(filename="hector_configuration.pdf")
    def download_plot():
        """Render the current committed configuration as a downloadable PDF."""
        result = state.get()
        if result is None:
            return None
        target_positions = pd.DataFrame(result.config["pos"]).reset_index(drop=True)
        guide_positions = pd.DataFrame(result.config["gpos"]).reset_index(drop=True)
        target_angles, guide_angles = _working_angles(result)
        flags = str(result.config.get("flags", "none"))
        fig, _ = make_field_figure(
            target_positions,
            target_angles,
            guide_positions,
            guide_angles,
            constants,
            standard_indices=result.config.get("standard_indices"),
            title=f"HECTOR configuration ({flags})",
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            path = Path(handle.name)
        fig.savefig(path)
        plt.close(fig)
        return str(path)


app = App(app_ui, server)


if __name__ == "__main__":  # pragma: no cover - manual launcher
    # ``shiny run`` starts the server but does not necessarily open a browser.
    # This module form is a convenient beginner-friendly alternative that
    # explicitly asks Shiny to launch the default browser.
    from shiny import run_app

    run_app("hectorconfig_py.app:app", launch_browser=True)
