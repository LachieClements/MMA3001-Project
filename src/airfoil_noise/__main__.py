"""Audit source data and create or verify frozen condition/fold assignments."""

import argparse
import json
from pathlib import Path

from airfoil_noise.data import SOURCE_SHA256, audit_data, load_airfoil
from airfoil_noise.split import read_frozen_splits, split_coverage, write_frozen_splits


DEFAULT_DATA = Path("data/raw/airfoil_self_noise.dat")
DEFAULT_SPLITS = Path("data/splits")


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main(argv: list[str] | None = None) -> int:
    """Run an explicit data/split operation without fitting an estimator.

    Parameters
    ----------
    argv : list of str, optional
        Arguments excluding executable name. None uses sys.argv.

    Returns
    -------
    int
        Zero after successful checks or generation.

    Raises
    ------
    ValueError
        Source integrity, schema or frozen-assignment checks fail.
    FileExistsError
        A requested output exists. Outputs are never overwritten.

    Notes
    -----
    Legacy baseline-cv is paused until Milestone 2 replaces its log-frequency
    transform and fold regeneration. Final evaluation is not implemented or
    invoked by these commands.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "make-split", "check-split"):
        part = sub.add_parser(command)
        part.add_argument("--data", type=Path, default=DEFAULT_DATA)
        part.add_argument("--expected-sha256", default=SOURCE_SHA256)
        if command == "audit":
            part.add_argument("--output", type=Path, help="Write a new JSON audit")
        else:
            part.add_argument("--directory", type=Path, default=DEFAULT_SPLITS)
        if command == "make-split":
            part.add_argument("--seed", type=int, default=3001)
            part.add_argument("--test-fraction", type=float, default=0.2)
            part.add_argument("--folds", type=int, default=5)
        if command == "check-split":
            part.add_argument("--coverage-output", type=Path, help="Write a new input-only coverage CSV")
    sub.add_parser("baseline-cv", help="Legacy command paused until Milestone 2")
    args = parser.parse_args(argv)
    if args.command == "baseline-cv":
        parser.error("baseline-cv is paused until Milestone 2 adopts raw inputs and frozen folds.")
    data = load_airfoil(args.data, expected_sha256=args.expected_sha256)
    if args.command == "audit":
        text = json.dumps(audit_data(data), indent=2, allow_nan=False) + "\n"
        if args.output:
            _write_new(args.output, text)
        print(text, end="")
        return 0
    if args.command == "make-split":
        metadata = write_frozen_splits(
            data, args.directory, seed=args.seed, test_fraction=args.test_fraction, n_splits=args.folds,
        )
    else:
        manifest, _, metadata = read_frozen_splits(data, args.directory)
        if args.coverage_output:
            _write_new(args.coverage_output, split_coverage(data, manifest).to_csv(index=False))
    print(json.dumps(metadata["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

