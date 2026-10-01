"""Command-line entry point for people who prefer a script to a notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

from .check import check_configuration
from .configure import configure_hector


def _configure_parser(subparsers):
    """Define the command-line options for one automatic configuration run."""
    parser = subparsers.add_parser("configure", help="configure a tile and guide CSV")
    parser.add_argument("--tile", required=True, help="input hexabundle/tile CSV")
    parser.add_argument("--guide", required=True, help="input guide-star CSV")
    parser.add_argument("--hexas-out", required=True, help="configured hexabundle CSV")
    parser.add_argument("--guides-out", required=True, help="configured guide CSV")
    parser.add_argument("--plot-out", help="optional diagnostic PDF or PNG")
    parser.add_argument("--constants", help="optional HECTOR_CONSTANTS.yaml")
    parser.add_argument("--quiet", action="store_true", help="suppress progress messages")
    parser.add_argument("--seed", type=int, default=2024, help="random seed for reproducible search")
    parser.add_argument(
        "--extended-samples-per-probe",
        type=int,
        default=400,
        help="random draws per implicated probe in the final hard-conflict search (default: 400)",
    )
    parser.add_argument(
        "--search-strategy",
        choices=("staged", "joint"),
        default="joint",
        help="use the joint Python solver (default) or the legacy R-like staged solver",
    )
    parser.add_argument(
        "--joint-candidate-pool",
        type=int,
        default=12,
        help="number of standard/guide candidates considered by the joint solver (default: 12)",
    )
    parser.add_argument(
        "--joint-beam-width",
        type=int,
        default=4,
        help="number of partial guide selections retained by the joint solver (default: 4)",
    )
    parser.add_argument(
        "--joint-samples-per-probe",
        type=int,
        default=80,
        help="random draws per implicated probe and angular scale in joint mode (default: 80)",
    )
    return parser


def _check_parser(subparsers):
    """Define the command-line options for checking configured CSVs."""
    parser = subparsers.add_parser("check", help="check configured tile and guide CSVs")
    parser.add_argument("--tile", required=True, help="configured hexabundle/tile CSV")
    parser.add_argument("--guide", required=True, help="configured guide-star CSV")
    parser.add_argument("--constants", help="optional HECTOR_CONSTANTS.yaml")
    return parser


def main(argv=None) -> int:
    """Run the CLI and return a shell exit status."""
    parser = argparse.ArgumentParser(prog="hector-config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _configure_parser(subparsers)
    _check_parser(subparsers)
    app_parser = subparsers.add_parser("app", help="print the Shiny app command")
    app_parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "app":
        if args.reload:
            print("Warning: omit --reload when the project contains .venv; it can repeatedly restart the server.")
        print("Run: shiny run --launch-browser hectorconfig_py.app:app")
        print("Alternative: python -m hectorconfig_py.app")
        return 0
    if args.command == "check":
        result = check_configuration(
            args.tile,
            args.guide,
            constants_path=args.constants,
        )
        print("\n".join(result.summary_lines()))
        if len(result.target_conflicts):
            print("\nTarget/standard conflict details:")
            print(result.target_conflicts.to_string(index=False))
        if len(result.guide_conflicts):
            print("\nGuide conflict details:")
            print(result.guide_conflicts.to_string(index=False))
        return 0 if result.ok else 2
    result = configure_hector(
        args.tile,
        args.guide,
        hexafile_out=args.hexas_out,
        guidefile_out=args.guides_out,
        plot_file=args.plot_out,
        constants_path=args.constants,
        seed=args.seed,
        extended_samples_per_probe=args.extended_samples_per_probe,
        search_strategy=args.search_strategy,
        joint_candidate_pool=args.joint_candidate_pool,
        joint_beam_width=args.joint_beam_width,
        joint_samples_per_probe=args.joint_samples_per_probe,
        verbose=not args.quiet,
    )
    print(f"Finished with flags: {result.config['flags']}")
    print(f"Hexas written to: {Path(args.hexas_out).resolve()}")
    print(f"Guides written to: {Path(args.guides_out).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
