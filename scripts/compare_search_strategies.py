"""Compare the joint and legacy staged HECTOR searches.

Run this from the ``hectorconfig_python`` directory after installing the
package.  It writes separate CSV/PDF outputs for each strategy and reports
runtime and final diagnostics side by side.  The two searches use the same
input field and random seed, but they are deliberately different algorithms,
so the comparison is about practical success and runtime rather than exact
numerical equality.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from hectorconfig_py import configure_hector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tile",
        default="src/hectorconfig_py/data/Hexas_G23_tile_263_NOT_CONFIGURED.csv",
    )
    parser.add_argument(
        "--guide",
        default="src/hectorconfig_py/data/Guides_G23_tile_263_NOT_CONFIGURED.csv",
    )
    parser.add_argument("--output-dir", default="example_output/strategy_comparison")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--joint-candidate-pool", type=int, default=12)
    parser.add_argument("--joint-beam-width", type=int, default=4)
    parser.add_argument("--joint-samples-per-probe", type=int, default=80)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for strategy in ("staged", "joint"):
        strategy_dir = output_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        result = configure_hector(
            args.tile,
            args.guide,
            hexafile_out=strategy_dir / "hexas_configured.csv",
            guidefile_out=strategy_dir / "guides_configured.csv",
            plot_file=strategy_dir / "configuration.pdf",
            seed=args.seed,
            search_strategy=strategy,
            joint_candidate_pool=args.joint_candidate_pool,
            joint_beam_width=args.joint_beam_width,
            joint_samples_per_probe=args.joint_samples_per_probe,
            verbose=True,
        )
        elapsed = perf_counter() - started
        print(f"\n{strategy}: {elapsed:.2f} seconds")
        print(f"  flags: {result.config['flags']}")
        print(f"  guides retained: {result.config['guide_count']}")
        print(f"  target/standard conflicts: {len(result.config['target_conflicts'])}")
        print(f"  guide conflicts: {len(result.config['guide_conflicts'])}")
        print(f"  target field valid: {result.config['target_fov']['ok']}")
        print(f"  guide field valid: {result.config['guide_fov']['ok']}")
        print(f"  output directory: {strategy_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
