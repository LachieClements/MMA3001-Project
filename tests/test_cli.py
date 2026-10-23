"""Small CLI workflows use synthetic data and never perform model selection."""

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


@pytest.mark.parametrize("command", ["baseline-cv", "final-evaluation", "evaluate-final"])
def test_model_and_final_evaluation_are_not_available(command: str) -> None:
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


def test_check_split_rejects_missing_bundle(tmp_path: Path, source_args: list[str]) -> None:
    with pytest.raises(ValueError, match="Incomplete"):
        main(["check-split", *source_args, "--directory", str(tmp_path / "missing")])
