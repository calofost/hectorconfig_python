# hectorconfig_python

This repository contains the Python distribution `hectorconfig-py`, imported
as `hectorconfig_py`. It is a translation of the HECTOR probe-configuration
workflow from the stable R package `hectorconfig` 0.1.18 in the separate
repository [`calofost/hectorconfig_R`](https://github.com/calofost/hectorconfig_R).
This checkout is Python version 0.1.10.

> **Important validation warning:** The Python implementation is still an
> early port, not a validated replacement for the R package. In testing, the
> Python solver performed substantially worse than the parent R
> `hectorconfig` 0.1.18 solver and required more manual intervention. Joint is
> the default here because it performed better than the alternative Python
> staged strategy on the test field; that is not evidence of parity with R.
> Before using a configuration for observing, run the same input through R and
> Python, compare the positions, angles, conflicts, field-of-view clearances,
> and plugability, and investigate discrepancies.

The Python version is intentionally released as a separate package while it is checked against the trusted R implementation. The default joint solver and the legacy staged solver share the R engine's geometry, acceptance checks, and file conventions, but their search paths differ:

1. read a distortion-corrected tile CSV and guide-star CSV;
2. select the 19 galaxy targets and two standard stars;
3. choose cable exit gaps and initial probe angles;
4. use bounded stochastic searches, including recursive retries, 20/45/90-degree passes, cable-exit-gap swaps, and 180-degree flip-plus-tweak starts;
5. in joint mode, optimise selected targets, standards, and guides through shared conflict components; in staged mode, clean and order candidates as the R engine does and then try candidates until six guides are configured;
6. allow implicated target-guide pairs to search together for target-guide clashes;
7. reject any final configuration whose complete probe polygon leaves or touches the 226 mm-radius field;
8. write configured CSV files and an optional diagnostic plot.

The Python Shiny app is a manual checker. It is not a replacement for the automatic search: it is where you inspect a difficult field, change a probe angle, re-check conflicts, and download the repaired tables.

## Install from this checkout

Use a virtual environment. This keeps Python package versions for HECTOR separate from the rest of your computer.

On macOS or Linux, from the `hectorconfig_python` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[app,dev]"
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[app,dev]"
```

The important distinction from R is that every command after activation must use the Python inside `.venv`. Check that with:

```bash
python --version
python -c "import hectorconfig_py; print(hectorconfig_py.__version__)"
```

## First test with the bundled example

Run this from the `hectorconfig_python` directory after activating the environment:

```bash
python -m hectorconfig_py.cli configure \
  --tile src/hectorconfig_py/data/Hexas_G23_tile_263_NOT_CONFIGURED.csv \
  --guide src/hectorconfig_py/data/Guides_G23_tile_263_NOT_CONFIGURED.csv \
  --hexas-out example_output/hexas_configured.csv \
  --guides-out example_output/guides_configured.csv \
  --plot-out example_output/configuration.pdf
```

To compare the default joint solver with the legacy staged solver on
the same field and seed, run:

```bash
python scripts/compare_search_strategies.py
```

This writes separate outputs under
`example_output/strategy_comparison/staged/` and
`example_output/strategy_comparison/joint/`, and reports runtime, flags,
remaining clashes, and field-of-view validity for each.

The output directory is created automatically. Look for:

```text
example_output/hexas_configured.csv
example_output/guides_configured.csv
example_output/configuration.pdf
```

The command prints a final `flags` value. `none` means that the automatic checks found no remaining conflict or field-of-view problem. `GuideFail`, `TargetFail`, `StdFail`, or `FieldFail` means that output was still written, but the field should be inspected in the app.

## Check configured output without running the solver

You can inspect output from either the R package or Python directly from the terminal:

```bash
python -m hectorconfig_py.cli check \
  --tile example_output/hexas_configured.csv \
  --guide example_output/guides_configured.csv
```

This reports the number of targets, standards, and guides; target/standard conflicts; guide conflicts; and complete-probe field-of-view failures. It exits with status `0` when the flags are `none`, and status `2` when manual intervention is required.

To reproduce a run exactly, use a fixed seed. The default is `2024`; you can make it explicit with `--seed 2024`.

For a particularly difficult target-guide clash, increase the final bounded
search budget, for example:

```bash
python -m hectorconfig_py.cli configure \
  --tile src/hectorconfig_py/data/Hexas_G23_tile_263_NOT_CONFIGURED.csv \
  --guide src/hectorconfig_py/data/Guides_G23_tile_263_NOT_CONFIGURED.csv \
  --hexas-out example_output/hexas_configured.csv \
  --guides-out example_output/guides_configured.csv \
  --plot-out example_output/configuration.pdf \
  --extended-samples-per-probe 500
```

The Python default is `400` draws per implicated probe. Increasing it can help
with a narrow valid angle interval, but increases runtime. The search remains
bounded and only varies the implicated target and guide probes.

## Run the Shiny app

Still in the activated environment and the `hectorconfig_python` directory:

```bash
shiny run --launch-browser hectorconfig_py.app:app
```

The `--launch-browser` option opens the default browser. If macOS does not
open it automatically, copy the local address printed by Shiny, normally
`http://127.0.0.1:8000`, into a browser. You can also use:

```bash
python -m hectorconfig_py.app
```

That form explicitly requests browser launch from Python.

For now, do not add `--reload` when running from this project directory: the
development reloader can watch the `.venv` package files and repeatedly restart
the server. Stop and restart the command manually after editing Python files.

The app opens without loading a field. Upload the raw tile and guide CSVs, then
click **Configure field**. If the uploaded tile already contains the configured
`probe`, `x`, `y`, and `angs` columns (from either the R or Python workflow),
the app loads it for manual checking instead of running the automatic solver
again.

The manual workflow is:

1. Double-click near a probe head in the plot. A red degree compass appears around the selected probe.
2. Single-click in the direction you want the probe tail/cable to point. The orange arrow previews the new angle.
3. Click **Fix Probe** to commit the preview, or enter an angle in degrees and then click **Fix Probe**. The automatic engine stores radians internally, but the app uses degrees for manual editing.
4. Click **List conflicted probes**. This checks target/standard clashes, guide clashes, and whether any complete polygon is outside the field.
5. Download the repaired hexas CSV, guide CSV, and current configuration PDF. The saved tables retain the original R-compatible column layout and can be loaded again.

Stop the app with `Ctrl-C` in the terminal.

## Use from Python code

This is the closest equivalent to the R example:

```python
from pathlib import Path

from hectorconfig_py import configure_hector

out = Path("example_output")
result = configure_hector(
    tile_file="src/hectorconfig_py/data/Hexas_G23_tile_263_NOT_CONFIGURED.csv",
    guide_file="src/hectorconfig_py/data/Guides_G23_tile_263_NOT_CONFIGURED.csv",
    hexafile_out=out / "hexas_configured.csv",
    guidefile_out=out / "guides_configured.csv",
    plot_file=out / "configuration.pdf",
    visualise=False,
)

print(result.config["flags"])
print(result.config["guide_count"])
print(result.config["guide_conflicts"])
```

### Joint search (default)

The default `search_strategy="joint"` selects a bounded pool of standards and
guides up front, then optimises targets, standards, and guides through shared
conflict components. This generally gives the Python port a better chance of
repairing a target angle when a later guide would otherwise clash with it:

```python
result = configure_hector(
    tile_file="src/hectorconfig_py/data/Hexas_G23_tile_263_NOT_CONFIGURED.csv",
    guide_file="src/hectorconfig_py/data/Guides_G23_tile_263_NOT_CONFIGURED.csv",
    joint_candidate_pool=12,
    joint_beam_width=4,
    joint_samples_per_probe=80,
)
print(result.config["joint_search"])
```

This search is bounded, so a different candidate set can be selected and a
hard field can still need manual repair. Increase `joint_samples_per_probe`
for a harder field, or reduce it while testing runtime. To use the legacy
R-like staged workflow explicitly, set `search_strategy="staged"` in Python or
pass `--search-strategy staged` on the command line. The command-line form of
the default joint workflow is:

```bash
python -m hectorconfig_py.cli configure \
  --tile src/hectorconfig_py/data/Hexas_G23_tile_263_NOT_CONFIGURED.csv \
  --guide src/hectorconfig_py/data/Guides_G23_tile_263_NOT_CONFIGURED.csv \
  --hexas-out example_output/hexas_joint.csv \
  --guides-out example_output/guides_joint.csv \
  --plot-out example_output/configuration_joint.pdf \
  --search-strategy joint
```

Useful objects are:

* `result.hexas`: configured tile/hexabundle table as a pandas DataFrame;
* `result.guides`: configured guide table as a pandas DataFrame;
* `result.config["angs"]`: target and standard angles in radians;
* `result.config["gangs"]`: guide angles in radians;
* `result.config["manual_intervention"]`: Boolean summary;
* `result.config["target_conflicts"]` and `result.config["guide_conflicts"]`: conflict tables;
* `result.config["target_fov"]` and `result.config["guide_fov"]`: field-of-view checks.

The Python equivalent of the R constants diagnostic is:

```python
from hectorconfig_py import hector_engine_parameters

parameters = hector_engine_parameters()
print(parameters["plate_radius"])
print(parameters["ceg_positions"])
```

## Run tests

After installing the `dev` extra:

```bash
python -m pytest -q
```

If `pytest` is not found, the usual cause is that the virtual environment is not active. Run `source .venv/bin/activate` again on macOS/Linux, or `.venv\Scripts\Activate.ps1` again in PowerShell.

## Translating R concepts to Python

| R | Python in this package |
|---|---|
| `data.frame` | `pandas.DataFrame` |
| `list` returned by `configure_hector()` | `ConfigurationResult` with `.config`, `.hexas`, and `.guides` |
| `nrow(x)` | `len(x)` |
| `x$column` | `x["column"]` or `x.column` for a pandas column |
| radians/degrees | NumPy radians internally; app input is degrees |
| `graphics`/`plot()` | Matplotlib |
| Shiny for R | Shiny for Python |
| `setwd()` and hard-coded paths | paths passed as function arguments or `pathlib.Path` |

## Install from GitHub

Once this repository is available at `calofost/hectorconfig_python`, users can
install it directly with:

```bash
python -m pip install "hectorconfig-py[app] @ git+https://github.com/calofost/hectorconfig_python.git"
```

## Current validation status

Version 0.1.10 makes the joint target/standard/guide solver the Python
default because it performed better than the Python staged strategy on the
test field; the legacy R-like staged solver remains available with
`search_strategy="staged"` or `--search-strategy staged`. This choice does
not indicate parity with R: in testing, the Python implementation performed
substantially worse than the parent R `hectorconfig` 0.1.18 solver and still
needs manual validation. The joint inner loop keeps stochastic scoring
bounded and applies wedge/plugability checks to accepted candidates and final
outputs. The app starts empty, uses one Configure field button, infers
preserved CEG choices when reopening legacy CSVs, and can download a current
PDF plot. Plots show science targets in blue, standard probes in purple, and
guide probes in green, with a legend. Manual angle diagnostics identify the
affected probe IDs and allowed CEG ranges. Neither stochastic path will be
numerically identical to R because NumPy and R use different random-number
generators. Before relying on any configuration for observing, run the same
input through R `hectorconfig` 0.1.18 and Python, compare positions, angles,
conflict tables, field-of-view clearances, and plugability, and investigate
any discrepancy rather than silently accepting it.

The diagnostic plot puts each target label or `G1`–`G6` label at the centre of its probe-head circle, with a small opaque background so the text remains readable over the probe geometry.

Saved hexabundle and guide files retain the R-compatible field layout. In
particular, the hexabundle file has one canonical `probe` column, and saving
the same file again does not add a second probe column.
