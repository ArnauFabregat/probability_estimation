import numpy as np
import pytest
import torch.nn as nn

from probability_prediction.model import MonotonicLinear
from probability_prediction.utils import (
    stochastic_baseline,
)


def test_stochastic_baseline_probability_handling():
    """
    Test stochastic_baseline handles probability clipping and edge cases correctly.
    """
    # Test with p < 0 (should be clipped to 0)
    cm_neg, p_neg, r_neg, f_neg = stochastic_baseline(p=-0.5, seed=42, verbose=False)
    assert p_neg == 0.0  # Precision should be 0 when all predictions are 0
    assert r_neg == 0.0  # Recall should be 0 when all predictions are 0
    assert f_neg == 0.0  # F1 should be 0 when all predictions are 0

    # Test with p > 1 (should be clipped to 1)
    cm_pos, p_pos, r_pos, f_pos = stochastic_baseline(p=1.5, seed=42, verbose=False)
    assert p_pos == 1.0  # Precision should be 1 when all predictions are 1
    assert r_pos == 1.0  # Recall should be 1 when all predictions are 1
    assert f_pos == 1.0  # F1 should be 1 when all predictions are 1

    # Test with very small p (close to 0)
    cm_small, p_small, r_small, f_small = stochastic_baseline(p=1e-10, seed=42, verbose=False)
    assert p_small < 0.1  # Should be very low precision
    assert r_small < 0.1  # Should be very low recall
    assert f_small < 0.1  # Should be very low F1

    # Test with very large p (close to 1)
    cm_large, p_large, r_large, f_large = stochastic_baseline(p=1 - 1e-10, seed=42, verbose=False)
    assert p_large > 0.9  # Should be very high precision
    assert r_large > 0.9  # Should be very high recall
    assert f_large > 0.9  # Should be very high F1

    # Test that clipping is consistent across calls
    cm1 = stochastic_baseline(p=1.5, seed=42, verbose=False)
    cm2 = stochastic_baseline(p=2.0, seed=42, verbose=False)
    assert np.allclose(cm1[0], cm2[0])  # Both should be clipped to p=1


def test_stochastic_baseline_reproducibility():
    """
    Test stochastic_baseline produces reproducible results with same seed.
    """
    # Test reproducibility with same seed
    cm1, p1, r1, f1 = stochastic_baseline(seed=42, verbose=False)
    cm2, p2, r2, f2 = stochastic_baseline(seed=42, verbose=False)

    assert np.allclose(cm1, cm2)
    assert p1 == p2
    assert r1 == r2
    assert f1 == f2

    # Test different seeds produce different results
    cm_diff, p_diff, r_diff, f_diff = stochastic_baseline(seed=123, verbose=False)

    assert not np.allclose(cm1, cm_diff)
    assert p1 != p_diff
    assert r1 != r_diff
    assert f1 != f_diff

    # Test that changing N affects results
    cm_n1, _, _, _ = stochastic_baseline(N=100, seed=42, verbose=False)
    cm_n2, _, _, _ = stochastic_baseline(N=200, seed=42, verbose=False)

    assert not np.allclose(cm_n1, cm_n2)

    # Test that changing p affects results
    cm_p1, _, _, _ = stochastic_baseline(p=0.3, seed=42, verbose=False)
    cm_p2, _, _, _ = stochastic_baseline(p=0.7, seed=42, verbose=False)

    assert not np.allclose(cm_p1, cm_p2)


def test_monotonic_linear_init_basic():
    """
    Test MonotonicLinear.__init__ with basic valid inputs.
    """
    # Test positive sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="+")
    assert model.in_features == 10
    assert model.out_features == 5
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test negative sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="-")
    assert model.sign == "-"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test default sign
    model = MonotonicLinear(in_features=10, out_features=5)
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)


def test_monotonic_linear_init_invalid_sign():
    """
    Test MonotonicLinear.__init__ with invalid sign values.
    """
    # Test invalid sign raises ValueError
    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="x")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="++")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign=None)


def test_monotonic_linear_init_parameter_shapes():
    """
    Test that raw_weight and bias parameters have correct shapes.
    """
    # Test various in/out feature combinations
    test_cases = [
        (10, 5),  # Regular case
        (1, 1),  # Minimal case
        (100, 50),  # Larger case
        (3, 7),  # Different dimensions
    ]

    for in_features, out_features in test_cases:
        # Test positive sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="+")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)

        # Test negative sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="-")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)


def test_monotonic_linear_init_basic():
    """
    Test MonotonicLinear.__init__ with basic valid inputs.
    """
    # Test positive sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="+")
    assert model.in_features == 10
    assert model.out_features == 5
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test negative sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="-")
    assert model.sign == "-"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test default sign
    model = MonotonicLinear(in_features=10, out_features=5)
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)


def test_monotonic_linear_init_invalid_sign():
    """
    Test MonotonicLinear.__init__ with invalid sign values.
    """
    # Test invalid sign raises ValueError
    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="x")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="++")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign=None)


def test_monotonic_linear_init_parameter_shapes():
    """
    Test that raw_weight and bias parameters have correct shapes.
    """
    # Test various in/out feature combinations
    test_cases = [
        (10, 5),  # Regular case
        (1, 1),  # Minimal case
        (100, 50),  # Larger case
        (3, 7),  # Different dimensions
    ]

    for in_features, out_features in test_cases:
        # Test positive sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="+")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)

        # Test negative sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="-")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)


def test_monotonic_linear_init_basic():
    """
    Test MonotonicLinear.__init__ with basic valid inputs.
    """
    # Test positive sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="+")
    assert model.in_features == 10
    assert model.out_features == 5
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test negative sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="-")
    assert model.sign == "-"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test default sign
    model = MonotonicLinear(in_features=10, out_features=5)
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)


def test_monotonic_linear_init_invalid_sign():
    """
    Test MonotonicLinear.__init__ with invalid sign values.
    """
    # Test invalid sign raises ValueError
    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="x")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="++")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign=None)


def test_monotonic_linear_init_parameter_shapes():
    """
    Test that raw_weight and bias parameters have correct shapes.
    """
    # Test various in/out feature combinations
    test_cases = [
        (10, 5),  # Regular case
        (1, 1),  # Minimal case
        (100, 50),  # Larger case
        (3, 7),  # Different dimensions
    ]

    for in_features, out_features in test_cases:
        # Test positive sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="+")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)

        # Test negative sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="-")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)


def test_monotonic_linear_init_basic():
    """
    Test MonotonicLinear.__init__ with basic valid inputs.
    """
    # Test positive sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="+")
    assert model.in_features == 10
    assert model.out_features == 5
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test negative sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="-")
    assert model.sign == "-"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test default sign
    model = MonotonicLinear(in_features=10, out_features=5)
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)


def test_monotonic_linear_init_invalid_sign():
    """
    Test MonotonicLinear.__init__ with invalid sign values.
    """
    # Test invalid sign raises ValueError
    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="x")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="++")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign=None)


def test_monotonic_linear_init_parameter_shapes():
    """
    Test that raw_weight and bias parameters have correct shapes.
    """
    # Test various in/out feature combinations
    test_cases = [
        (10, 5),  # Regular case
        (1, 1),  # Minimal case
        (100, 50),  # Larger case
        (3, 7),  # Different dimensions
    ]

    for in_features, out_features in test_cases:
        # Test positive sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="+")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)

        # Test negative sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="-")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)


def test_monotonic_linear_init_basic():
    """
    Test MonotonicLinear.__init__ with basic valid inputs.
    """
    # Test positive sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="+")
    assert model.in_features == 10
    assert model.out_features == 5
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test negative sign
    model = MonotonicLinear(in_features=10, out_features=5, sign="-")
    assert model.sign == "-"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)

    # Test default sign
    model = MonotonicLinear(in_features=10, out_features=5)
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (5, 10)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (5,)


def test_monotonic_linear_init_invalid_sign():
    """
    Test MonotonicLinear.__init__ with invalid sign values.
    """
    # Test invalid sign raises ValueError
    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="x")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="++")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign="")

    with pytest.raises(ValueError, match="sign must be '\+' or '-'"):
        MonotonicLinear(in_features=10, out_features=5, sign=None)


def test_monotonic_linear_init_parameter_shapes():
    """
    Test that raw_weight and bias parameters have correct shapes.
    """
    # Test various in/out feature combinations
    test_cases = [
        (10, 5),  # Regular case
        (1, 1),  # Minimal case
        (100, 50),  # Larger case
        (3, 7),  # Different dimensions
    ]

    for in_features, out_features in test_cases:
        # Test positive sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="+")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)

        # Test negative sign
        model = MonotonicLinear(in_features=in_features, out_features=out_features, sign="-")
        assert model.raw_weight.shape == (out_features, in_features)
        assert model.bias.shape == (out_features,)
