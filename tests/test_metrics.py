"""Hand-calculated errors and aggregation with unequal condition sizes."""

import numpy as np
import pytest

from airfoil_noise.metrics import condition_metrics, pooled_metrics, regression_metrics


def test_hand_calculated_pooled_metrics() -> None:
    result = pooled_metrics([1, 2, 4], [2, 2, 2])
    assert result["mae_db"] == pytest.approx(1)
    assert result["rmse_db"] == pytest.approx(np.sqrt(5 / 3))
    assert result["r2"] == pytest.approx(-1 / 14)
    assert result["r2_undefined_reason"] is None


def test_perfect_predictions() -> None:
    result = regression_metrics([1, 3, 8], [1, 3, 8], ["a", "a", "b"])
    assert result["mae_db"] == result["rmse_db"] == result["equal_condition_mae_db"] == 0
    assert result["r2"] == pytest.approx(1)


def test_unequal_condition_sizes() -> None:
    truth, prediction, groups = [0, 0, 0, 0], [1, 1, 1, 9], ["a", "a", "a", "b"]
    result = regression_metrics(truth, prediction, groups)
    assert result["mae_db"] == pytest.approx(3)
    assert result["equal_condition_mae_db"] == pytest.approx(5)
    per_condition = condition_metrics(truth, prediction, groups)
    assert per_condition["n_observations"].tolist() == [3, 1]
    assert per_condition["mae_db"].tolist() == pytest.approx([1, 9])
    assert per_condition["rmse_db"].tolist() == pytest.approx([1, 9])


@pytest.mark.parametrize("truth,prediction,reason", [
    ([2], [2], "fewer_than_two_observations"),
    ([2, 2], [2, 2], "constant_truth"),
    ([2, 2], [10, 10], "constant_truth"),
])
def test_undefined_r2(truth, prediction, reason) -> None:
    result = pooled_metrics(truth, prediction)
    assert np.isnan(result["r2"])
    assert result["r2_undefined_reason"] == reason


@pytest.mark.parametrize("truth,prediction", [
    ([], []), ([1, 2], [1]), ([[1], [2]], [1, 2]), ([1, 2], [[1], [2]]),
    ([1, np.nan], [1, 2]), ([1, 2], [1, np.inf]), ([1, -np.inf], [1, 2]),
    (["1"], [1]), ([1], ["1"]), ([True], [1]), ([1], [1j]),
])
def test_invalid_measurements_rejected(truth, prediction) -> None:
    with pytest.raises(ValueError):
        pooled_metrics(truth, prediction)
    with pytest.raises(ValueError):
        condition_metrics(truth, prediction, ["a"])


@pytest.mark.parametrize("groups", [[], ["a"], [["a"], ["b"]], ["a", None], ["a", ""], ["a", 1]])
def test_invalid_condition_identifiers_rejected(groups) -> None:
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1, 2], groups)
