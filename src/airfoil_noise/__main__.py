"""Command-line entry points for data checks and development baselines."""

import argparse
from pathlib import Path

from airfoil_noise.baseline import development_baselines
from airfoil_noise.data import audit_data, load_airfoil
from airfoil_noise.split import apply_manifest, make_manifest, read_manifest


DEFAULT_DATA = Path("data/raw/airfoil_self_noise.dat")
DEFAULT_MANIFEST = Path("data/splits/conditions.csv")


def main(argv: list[str] | None = None) -> int:
    """Run an audit, create/check a fixed split, or compare baselines.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments without executable name. ``None`` uses the
        current process arguments. Paths default to the repository layout.

    Returns
    -------
    int
        Zero after a successful command. Invalid inputs raise exceptions or
        are rejected by ``argparse``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "make-split", "check-split", "baseline-cv"):
        part = sub.add_parser(command)
        part.add_argument("--data", type=Path, default=DEFAULT_DATA)
        if command != "audit":
            part.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        if command == "make-split":
            part.add_argument(
                "--force", action="store_true", help="Replace an existing split"
            )
        if command == "baseline-cv":
            part.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    data = load_airfoil(args.data)

    if args.command == "audit":
        summary = audit_data(data)
        for key in ("rows", "exact_duplicate_rows", "conditions", "smallest_sweep", "largest_sweep"):
            print(f"{key}: {summary[key]}")
        for key, (smallest, largest) in summary["ranges"].items():
            print(f"{key}: [{smallest:g}, {largest:g}]")
        return 0

    if args.command == "make-split":
        if args.manifest.exists() and not args.force:
            raise FileExistsError(
                f"{args.manifest} already exists; use --force only if re-splitting is intentional."
            )
        manifest = make_manifest(data)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.manifest, index=False)
        print(f"Wrote {len(manifest)} conditions to {args.manifest}")
        return 0

    manifest = read_manifest(args.manifest)
    development, test = apply_manifest(data, manifest)
    if args.command == "check-split":
        print(
            f"development: {len(development)} rows in "
            f"{(manifest['split'] == 'development').sum()} conditions"
        )
        print(
            f"test: {len(test)} rows in "
            f"{(manifest['split'] == 'test').sum()} conditions"
        )
        return 0

    results = development_baselines(development, n_splits=args.folds)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nDevelopment-fold means:")
    print(
        results.groupby("model")[["mae_db", "rmse_db", "fit_predict_seconds"]]
        .mean()
        .to_string(float_format=lambda value: f"{value:.4f}")
    )
    print("\nHeld-out test labels were not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

