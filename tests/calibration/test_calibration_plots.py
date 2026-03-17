import pytest
import torch.nn as nn

from probability_prediction.model import MonotonicLinear


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
