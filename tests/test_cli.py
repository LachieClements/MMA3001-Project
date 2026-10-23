"""CLI checks and small explicitly synthetic workflows."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from airfoil_noise.__main__ import main
from airfoil_noise.data import COLUMNS, file_sha256


@pytest.fixture
def source_args(tmp_path: Path) -> list[str]:
    source = tmp_path / "synthetic.dat"
    rows = [
        [frequency, angle, 0.1, 40, 0.002, 120 + angle]
        for angle in range(15)
        for frequency in [500, 1000, 2000]
    ]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(source, sep="\t", header=False, index=False)
    return ["--data", str(source), "--expected-sha256", file_sha256(source)]


def test_small_workflow_has_traceable_outputs_and_no_fits(
    tmp_path: Path, source_args: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Data/split commands must not fit an estimator.")

    for model in (DummyRegressor, LinearRegression, DecisionTreeRegressor, SVR, MLPRegressor):
        monkeypatch.setattr(model, "fit", forbidden_fit)
    source = Path(source_args[1])
    original_bytes = source.read_bytes()
    audit = tmp_path / "audit.json"
    directory = tmp_path / "splits"
    coverage = tmp_path / "coverage.csv"
    assert main(["audit", *source_args, "--output", str(audit)]) == 0
    assert main(["make-split", *source_args, "--directory", str(directory)]) == 0
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert main(["check-split", *source_args, "--directory", str(directory),
                 "--coverage-output", str(coverage)]) == 0
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert json.loads(audit.read_text(encoding="utf-8"))["source_sha256"] == file_sha256(source)
    assert metadata["counts"]["development"]["rows"] + metadata["counts"]["test"]["rows"] == 45
    assert len(pd.read_csv(coverage)) == 10
    assert source.read_bytes() == original_bytes
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert set(p.name for p in tmp_path.iterdir()) == {
        "synthetic.dat", "audit.json", "splits", "coverage.csv",
    }


@pytest.mark.parametrize("operation", ["audit", "make-split", "coverage"])
def test_output_replacement_is_rejected(
    tmp_path: Path, source_args: list[str], operation: str,
) -> None:
    directory = tmp_path / "splits"
    if operation == "make-split":
        arguments = ["make-split", *source_args, "--directory", str(directory)]
        main(arguments)
        target = directory / "metadata.json"
    else:
        target = tmp_path / "record"
        if operation == "audit":
            arguments = ["audit", *source_args, "--output", str(target)]
        else:
            main(["make-split", *source_args, "--directory", str(directory)])
            arguments = ["check-split", *source_args, "--directory", str(directory),
                         "--coverage-output", str(target)]
        main(arguments)
    previous = target.read_bytes()
    with pytest.raises(FileExistsError):
        main(arguments)
    assert target.read_bytes() == previous


def test_custom_source_requires_declared_checksum(source_args: list[str]) -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main(["audit", *source_args[:2]])


@pytest.mark.parametrize("command", ["final-evaluation", "evaluate-final"])
def test_final_evaluation_is_not_available(command: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2


def test_module_help_command_works(tmp_path: Path) -> None:
    """Verify the installed package from outside the source checkout."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "airfoil_noise", "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "check-split" in result.stdout


@pytest.mark.parametrize("command", ["development-study", "benchmark-development"])
def test_study_commands_require_explicit_output(command):
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2


def test_synthetic_study_cli_uses_saved_assignments(tmp_path, source_args, monkeypatch):
    from copy import deepcopy
    protocol = json.loads(Path("configs/milestone4.json").read_text())
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    # Keep this CLI test small; all-five-model paths are checked separately.
    protocol["families"] = protocol["families"][:2]
    protocol.update(run_kind="synthetic_cli_check", expected_development_rows=36,
                    expected_development_conditions=12, configuration_count=2,
                    variant_count=2, cv_fit_budget=10, total_fit_budget=22)
    protocol["benchmark"].update(batch_rows=16, single_rows=4, blocks=2,
                                 minimum_block_seconds=0, prediction_warmup_rounds=1)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    split = tmp_path / "split"
    main(["make-split", *source_args, "--directory", str(split)])
    before = {path.name: path.read_bytes() for path in split.iterdir()}
    search, benchmark = tmp_path / "search", tmp_path / "benchmark"
    common = [*source_args, "--directory", str(split), "--protocol", str(protocol_path)]
    assert main(["development-study", *common, "--output-dir", str(search)]) == 0
    assert main(["benchmark-development", *common, "--selection",
                 str(search / "selected_configurations.json"), "--output-dir", str(benchmark)]) == 0
    assert json.loads((search / "search.json").read_text())["fits_attempted"] == 10
    assert json.loads((benchmark / "benchmark.json").read_text())["fits_attempted"] == 12
    assert before == {path.name: path.read_bytes() for path in split.iterdir()}
    with pytest.raises(FileExistsError):
        main(["development-study", *common, "--output-dir", str(search)])


def test_check_split_rejects_missing_bundle(tmp_path: Path, source_args: list[str]) -> None:
    with pytest.raises(ValueError, match="Incomplete"):
        main(["check-split", *source_args, "--directory", str(tmp_path / "missing")])


def test_baseline_cli_uses_frozen_folds_and_saves_traceable_records(tmp_path: Path, source_args: list[str]) -> None:
    directory, output = tmp_path / "splits", tmp_path / "development-check"
    main(["make-split", *source_args, "--directory", str(directory)])
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    arguments = ["baseline-cv", *source_args, "--directory", str(directory), "--output-dir", str(output)]
    assert main(arguments) == 0
    run = json.loads((output / "run.json").read_text())
    predictions = pd.read_csv(output / "predictions.csv")
    assert run["kind"] == "development_cross_validation"
    assert run["fits_attempted"] == 10 and run["fits_failed"] == 0
    assert len(predictions) == 2 * run["n_observations"]
    manifest = pd.read_csv(directory / "conditions.csv")
    test_ids = manifest.loc[manifest["split"] == "test", "condition_id"]
    assert set(predictions["condition_id"]).isdisjoint(test_ids)
    assert not predictions.duplicated(["model", "row_id"]).any()
    assert run["provenance"]["source_sha256"] == source_args[3]
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(FileExistsError):
        main(arguments)


def test_baseline_cli_requires_explicit_output_directory() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["baseline-cv"])
    assert exc.value.code == 2


def test_baseline_cli_refuses_missing_folds(tmp_path: Path, source_args: list[str]) -> None:
    output = tmp_path / "development-check"
    with pytest.raises(ValueError, match="Incomplete"):
        main(["baseline-cv", *source_args, "--directory", str(tmp_path / "missing"), "--output-dir", str(output)])
    assert not output.exists()
