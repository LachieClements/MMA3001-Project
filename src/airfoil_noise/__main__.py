"""Audit frozen data and explicitly run development-only studies."""

import argparse
import json
from pathlib import Path

from airfoil_noise.baseline import development_baselines
from airfoil_noise.data import SOURCE_SHA256, audit_data, file_sha256, load_airfoil
from airfoil_noise.split import load_development, read_frozen_splits, split_coverage, write_frozen_splits
from airfoil_noise.validation import save_development_results


DEFAULT_DATA = Path("data/raw/airfoil_self_noise.dat")
DEFAULT_SPLITS = Path("data/splits")


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main(argv: list[str] | None = None) -> int:
    """Run an explicit data/split check or development-only baseline study.

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
    baseline-cv, development-study and benchmark-development explicitly fit.
    Final evaluation is not implemented or invoked by these commands.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    figures = sub.add_parser("plot-development")
    figures.add_argument("--search-dir", type=Path, required=True)
    figures.add_argument("--benchmark-dir", type=Path, required=True)
    figures.add_argument("--output-dir", type=Path, required=True)
    for command in ("audit", "make-split", "check-split", "baseline-cv", "development-study", "benchmark-development"):
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
        if command in ("baseline-cv", "development-study", "benchmark-development"):
            part.add_argument("--output-dir", type=Path, required=True, help="New development run directory")
        if command in ("development-study", "benchmark-development"):
            part.add_argument("--protocol", type=Path, default=Path("configs/milestone4.json"))
        if command == "benchmark-development":
            part.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "output_dir") and args.output_dir.exists():
        raise FileExistsError(f"Refusing to replace {args.output_dir}.")
    if args.command == "plot-development":
        from airfoil_noise.plotting import plot_development
        print(plot_development(args.search_dir, args.benchmark_dir, args.output_dir))
        return 0
    data = load_airfoil(args.data, expected_sha256=args.expected_sha256)
    if args.command in ("development-study", "benchmark-development"):
        from airfoil_noise.development import run_study
        from airfoil_noise.benchmarking import benchmark_selected
        _, _, metadata = read_frozen_splits(data, args.directory)
        development, row_folds = load_development(data, args.directory)
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        provenance = {
            "source_path": str(args.data), "source_sha256": data.attrs["source_sha256"],
            "split_directory": str(args.directory), "frozen_split_metadata": metadata,
            "protocol_path": str(args.protocol), "protocol_sha256": file_sha256(args.protocol),
            "implementation_sha256": {
                path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
            },
        }
        if args.command == "development-study":
            record = run_study(development, row_folds, protocol, args.output_dir, provenance=provenance)
            print(f'Frozen accuracy leader: {record["accuracy_leader"]}')
        else:
            record = benchmark_selected(development, args.selection, protocol, args.output_dir, provenance=provenance)
            print(json.dumps(record, indent=2))
        return 0
    if args.command == "audit":
        text = json.dumps(audit_data(data), indent=2, allow_nan=False) + "\n"
        if args.output:
            _write_new(args.output, text)
        print(text, end="")
        return 0
    if args.command == "baseline-cv":
        # The partition boundary owns the full source. Evaluators receive only
        # development rows and their saved mapping, never final-test targets.
        _, _, metadata = read_frozen_splits(data, args.directory)
        development, row_folds = load_development(data, args.directory)
        results = development_baselines(development, row_folds)
        save_development_results(results, args.output_dir, provenance={
            "source_path": str(args.data), "source_sha256": data.attrs["source_sha256"],
            "split_directory": str(args.directory), "frozen_split_metadata": metadata,
            "implementation_sha256": {
                name: file_sha256(Path(__file__).with_name(name)) for name in (
                    "data.py", "split.py", "models.py", "metrics.py", "validation.py", "baseline.py", "__main__.py",
                )
            },
        })
        if results["status"] != "success":
            raise RuntimeError(f"Development evaluation failed; inspect {args.output_dir}/diagnostics.json.")
        print(results["summary"].to_string(index=False))
        print(f"Development records saved to {args.output_dir}. Timings are preliminary diagnostics.")
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

