"""Numpy-to-native conversion and constraint clipping of sampled values.

``DistributionSampler._convert_numpy_value`` turns raw numpy draws into native
Python values and clips them to any ``ge``/``le``/``gt``/``lt`` constraints
declared on the field (via ``Field(...)``). This is what keeps sampled numbers
inside their model's bounds before Pydantic validation runs.
"""

import numpy as np

from gendantic.sampler import DistributionSampler

NO_CONSTRAINTS: dict[str, float | None] = {
    "ge": None,
    "le": None,
    "gt": None,
    "lt": None,
}


def _sampler() -> DistributionSampler:
    return DistributionSampler(seed=0)


def test_numpy_integer_becomes_python_int() -> None:
    result = _sampler()._convert_numpy_value(np.int64(7), int, None)
    assert result == 7
    assert isinstance(result, int)


def test_numpy_float_becomes_python_float() -> None:
    result = _sampler()._convert_numpy_value(np.float64(3.5), float, None)
    assert result == 3.5
    assert isinstance(result, float)


def test_numpy_array_becomes_list() -> None:
    result = _sampler()._convert_numpy_value(np.array([1, 2, 3]))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_float_cast_to_int_rounds() -> None:
    assert _sampler()._convert_numpy_value(np.float64(3.7), int, None) == 4


def test_ge_clamps_lower_bound() -> None:
    result = _sampler()._convert_numpy_value(
        np.float64(-5.0), float, {**NO_CONSTRAINTS, "ge": 0.0}
    )
    assert result == 0.0


def test_le_clamps_upper_bound() -> None:
    result = _sampler()._convert_numpy_value(
        np.float64(150.0), int, {**NO_CONSTRAINTS, "le": 100.0}
    )
    assert result == 100


def test_gt_pushes_above_bound_for_int() -> None:
    # Strictly-greater on an int field steps up by 1.
    result = _sampler()._convert_numpy_value(
        np.int64(5), int, {**NO_CONSTRAINTS, "gt": 5.0}
    )
    assert result == 6


def test_gt_pushes_above_bound_for_float() -> None:
    result = _sampler()._convert_numpy_value(
        np.float64(5.0), float, {**NO_CONSTRAINTS, "gt": 5.0}
    )
    assert result > 5.0


def test_lt_pushes_below_bound_for_float() -> None:
    result = _sampler()._convert_numpy_value(
        np.float64(1.0), float, {**NO_CONSTRAINTS, "lt": 1.0}
    )
    assert result < 1.0


def test_value_within_bounds_is_untouched() -> None:
    result = _sampler()._convert_numpy_value(
        np.float64(50.0), float, {"ge": 0.0, "le": 100.0, "gt": None, "lt": None}
    )
    assert result == 50.0


def test_non_numpy_value_passes_through() -> None:
    # A plain Python value (not a numpy scalar/array) is returned unchanged.
    assert _sampler()._convert_numpy_value("hello", str, None) == "hello"
