import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.init as init

from probability_prediction.model import MonotonicLinear, MonotonicNN


def test_monotonic_linear_init_valid_inputs():
    """
    Test MonotonicLinear.__init__ with valid inputs for both signs.
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
    model = MonotonicLinear(in_features=3, out_features=7, sign="-")
    assert model.in_features == 3
    assert model.out_features == 7
    assert model.sign == "-"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (7, 3)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (7,)

    # Test default sign
    model = MonotonicLinear(in_features=4, out_features=2)
    assert model.sign == "+"
    assert isinstance(model.raw_weight, nn.Parameter)
    assert model.raw_weight.shape == (2, 4)
    assert isinstance(model.bias, nn.Parameter)
    assert model.bias.shape == (2,)


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


def test_monotonic_nn_init_basic():
    """
    Test MonotonicNN.__init__ with basic configurations.
    """
    # Test single branch: non-monotonic only
    model = MonotonicNN(all_variables=["x1", "x2", "x3"], non_monotonic_vars=["x1", "x2"], hidden_non=16)
    assert model.all_variables == ["x1", "x2", "x3"]
    assert model.non_monotonic_vars == ["x1", "x2"]
    assert model.positive_monotonic_vars == []
    assert model.negative_monotonic_vars == []
    assert model.mask_non is not None
    assert model.mask_pos is None
    assert model.mask_neg is None
    assert model.lin_non is not None
    assert model.lin_pos is None
    assert model.lin_neg is None
    assert model.out_non is not None
    assert model.out_pos is None
    assert model.out_neg is None

    # Test single branch: positive monotonic only
    model = MonotonicNN(all_variables=["x1", "x2", "x3"], positive_monotonic_vars=["x1", "x2"], hidden_pos=8)
    assert model.positive_monotonic_vars == ["x1", "x2"]
    assert model.mask_pos is not None
    assert model.lin_pos is not None
    assert isinstance(model.lin_pos, MonotonicLinear)
    assert model.lin_pos.sign == "+"
    assert model.out_pos is not None
    assert isinstance(model.out_pos, MonotonicLinear)
    assert model.out_pos.sign == "+"

    # Test single branch: negative monotonic only
    model = MonotonicNN(all_variables=["x1", "x2", "x3"], negative_monotonic_vars=["x1", "x2"], hidden_neg=8)
    assert model.negative_monotonic_vars == ["x1", "x2"]
    assert model.mask_neg is not None
    assert model.lin_neg is not None
    assert isinstance(model.lin_neg, MonotonicLinear)
    assert model.lin_neg.sign == "-"
    assert model.out_neg is not None
    assert isinstance(model.out_neg, MonotonicLinear)
    assert model.out_neg.sign == "+"  # Output layer is always positive

    # Test all three branches
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3", "x4", "x5"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2", "x3"],
        negative_monotonic_vars=["x4", "x5"],
        hidden_non=16,
        hidden_pos=8,
        hidden_neg=8,
    )
    assert model.mask_non is not None
    assert model.mask_pos is not None
    assert model.mask_neg is not None
    assert model.lin_non is not None
    assert model.lin_pos is not None
    assert model.lin_neg is not None
    assert model.out_non is not None
    assert model.out_pos is not None
    assert model.out_neg is not None


def test_monotonic_nn_init_branch_widths():
    """
    Test that branch widths are correctly handled, including zero widths.
    """
    # Test with zero width for non-monotonic branch
    model = MonotonicNN(all_variables=["x1", "x2"], non_monotonic_vars=["x1", "x2"], hidden_non=0)
    assert model.lin_non is None
    assert model.out_non is None

    # Test with zero width for positive monotonic branch
    model = MonotonicNN(all_variables=["x1", "x2"], positive_monotonic_vars=["x1", "x2"], hidden_pos=0)
    assert model.lin_pos is None
    assert model.out_pos is None

    # Test with zero width for negative monotonic branch
    model = MonotonicNN(all_variables=["x1", "x2"], negative_monotonic_vars=["x1", "x2"], hidden_neg=0)
    assert model.lin_neg is None
    assert model.out_neg is None

    # Test with zero width for all branches (edge case)
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3"],
        hidden_non=0,
        hidden_pos=0,
        hidden_neg=0,
    )
    assert model.lin_non is None
    assert model.lin_pos is None
    assert model.lin_neg is None
    assert model.out_non is None
    assert model.out_pos is None
    assert model.out_neg is None

    # Test with non-zero widths
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3"],
        hidden_non=5,
        hidden_pos=3,
        hidden_neg=4,
    )
    assert model.lin_non is not None
    assert model.lin_non.in_features == 1
    assert model.lin_non.out_features == 5
    assert model.lin_pos is not None
    assert model.lin_pos.in_features == 1
    assert model.lin_pos.out_features == 3
    assert model.lin_neg is not None
    assert model.lin_neg.in_features == 1
    assert model.lin_neg.out_features == 4


def test_monotonic_nn_init_weight_initialization():
    """
    Test that _init_weights() is called and weights are initialized.
    """
    # Create model and check that weights exist
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3"],
        hidden_non=4,
        hidden_pos=3,
        hidden_neg=2,
    )

    # Verify that all layers have weights initialized
    if model.lin_non:
        assert model.lin_non.weight is not None
        assert model.lin_non.bias is not None
    if model.lin_pos:
        assert model.lin_pos.raw_weight is not None
        assert model.lin_pos.bias is not None
    if model.lin_neg:
        assert model.lin_neg.raw_weight is not None
        assert model.lin_neg.bias is not None
    if model.out_non:
        assert model.out_non.weight is not None
        assert model.out_non.bias is not None
    if model.out_pos:
        assert model.out_pos.raw_weight is not None
        assert model.out_pos.bias is not None
    if model.out_neg:
        assert model.out_neg.raw_weight is not None
        assert model.out_neg.bias is not None

    # Verify weight initialization method (Xavier uniform)
    # This checks that weights are not all zero and have reasonable scale
    if model.lin_non:
        assert not torch.allclose(model.lin_non.weight, torch.zeros_like(model.lin_non.weight))
        assert model.lin_non.weight.std() > 0.01
    if model.lin_pos:
        assert not torch.allclose(model.lin_pos.raw_weight, torch.zeros_like(model.lin_pos.raw_weight))
        assert model.lin_pos.raw_weight.std() > 0.01
    if model.out_pos:
        assert not torch.allclose(model.out_pos.raw_weight, torch.zeros_like(model.out_pos.raw_weight))
        assert model.out_pos.raw_weight.std() > 0.01


def test_monotonic_nn_init_edge_cases():
    """
    Test MonotonicNN.__init__ with edge cases and unusual inputs.
    """
    # Test single variable in all_variables
    model = MonotonicNN(all_variables=["x1"], non_monotonic_vars=["x1"], hidden_non=1)
    assert model.mask_non is not None
    assert model.mask_non.shape == torch.Size([1])
    assert model.mask_non.item() == 0  # Only index 0

    # Test very large number of variables
    many_vars = [f"x{i}" for i in range(1000)]
    model = MonotonicNN(
        all_variables=many_vars,
        non_monotonic_vars=many_vars[:500],
        positive_monotonic_vars=many_vars[500:750],
        negative_monotonic_vars=many_vars[750:],
        hidden_non=64,
        hidden_pos=32,
        hidden_neg=32,
    )
    assert model.mask_non is not None
    assert model.mask_pos is not None
    assert model.mask_neg is not None
    assert model.mask_non.shape == torch.Size([500])
    assert model.mask_pos.shape == torch.Size([250])
    assert model.mask_neg.shape == torch.Size([250])

    # Test with boolean variable names (edge case)
    model = MonotonicNN(
        all_variables=["True", "False", "x"], non_monotonic_vars=["True", "False"], positive_monotonic_vars=["x"]
    )
    assert model.mask_non is not None
    assert model.mask_pos is not None

    # Test with numeric string variable names
    model = MonotonicNN(all_variables=["1", "2", "3"], non_monotonic_vars=["1", "2"], negative_monotonic_vars=["3"])
    assert model.mask_non is not None
    assert model.mask_neg is not None

    # Test with special characters in variable names
    model = MonotonicNN(
        all_variables=["x_1", "x-2", "x@3"], positive_monotonic_vars=["x_1", "x-2"], negative_monotonic_vars=["x@3"]
    )
    assert model.mask_pos is not None
    assert model.mask_neg is not None


def test_monotonic_nn_init_weights_basic():
    """
    Test _init_weights initializes all layers correctly.
    """
    # Create model with all branch types
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3", "x4"],
        non_monotonic_vars=["x1", "x2"],
        positive_monotonic_vars=["x3"],
        negative_monotonic_vars=["x4"],
        hidden_non=3,
        hidden_pos=2,
        hidden_neg=2,
    )

    # Call the method directly
    model._init_weights()

    # Verify standard Linear layers are initialized
    if model.lin_non:
        assert model.lin_non.weight is not None
        assert model.lin_non.bias is not None
        assert not torch.allclose(model.lin_non.weight, torch.zeros_like(model.lin_non.weight))
        assert torch.allclose(model.lin_non.bias, torch.zeros_like(model.lin_non.bias))

    if model.out_non:
        assert model.out_non.weight is not None
        assert model.out_non.bias is not None
        assert not torch.allclose(model.out_non.weight, torch.zeros_like(model.out_non.weight))
        assert torch.allclose(model.out_non.bias, torch.zeros_like(model.out_non.bias))

    # Verify MonotonicLinear layers are initialized
    if model.lin_pos:
        assert model.lin_pos.raw_weight is not None
        assert model.lin_pos.bias is not None
        assert not torch.allclose(model.lin_pos.raw_weight, torch.zeros_like(model.lin_pos.raw_weight))
        assert torch.allclose(model.lin_pos.bias, torch.zeros_like(model.lin_pos.bias))

    if model.lin_neg:
        assert model.lin_neg.raw_weight is not None
        assert model.lin_neg.bias is not None
        assert not torch.allclose(model.lin_neg.raw_weight, torch.zeros_like(model.lin_neg.raw_weight))
        assert torch.allclose(model.lin_neg.bias, torch.zeros_like(model.lin_neg.bias))

    if model.out_pos:
        assert model.out_pos.raw_weight is not None
        assert model.out_pos.bias is not None
        assert not torch.allclose(model.out_pos.raw_weight, torch.zeros_like(model.out_pos.raw_weight))
        assert torch.allclose(model.out_pos.bias, torch.zeros_like(model.out_pos.bias))

    if model.out_neg:
        assert model.out_neg.raw_weight is not None
        assert model.out_neg.bias is not None
        assert not torch.allclose(model.out_neg.raw_weight, torch.zeros_like(model.out_neg.raw_weight))
        assert torch.allclose(model.out_neg.bias, torch.zeros_like(model.out_neg.bias))


def test_monotonic_nn_init_weights_no_bias():
    """
    Test _init_weights handles layers with no bias correctly.
    """
    # Create model with some layers that might have no bias
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3", "x4"],
        non_monotonic_vars=["x1", "x2"],
        positive_monotonic_vars=["x3"],
        negative_monotonic_vars=["x4"],
        hidden_non=0,  # Should create None for lin_non and out_non
        hidden_pos=0,  # Should create None for lin_pos and out_pos
        hidden_neg=0,  # Should create None for lin_neg and out_neg
    )

    # Call initialization - should handle None gracefully
    model._init_weights()

    # Verify that None layers don't cause errors
    assert model.lin_non is None
    assert model.out_non is None
    assert model.lin_pos is None
    assert model.out_pos is None
    assert model.lin_neg is None
    assert model.out_neg is None

    # Test with bias=False in some layers (if supported)
    # Create a custom model to test bias=False scenario
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=False)
            self.monotonic = MonotonicLinear(3, 4, sign="+")

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                if isinstance(m, MonotonicLinear):
                    nn.init.xavier_uniform_(m.raw_weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    test_model = TestModel()
    test_model._init_weights()

    # Verify bias-less layer was initialized correctly
    assert test_model.linear.bias is None
    assert test_model.linear.weight is not None
    assert not torch.allclose(test_model.linear.weight, torch.zeros_like(test_model.linear.weight))

    # Verify MonotonicLinear with bias
    assert test_model.monotonic.raw_weight is not None
    assert test_model.monotonic.bias is not None
    assert not torch.allclose(test_model.monotonic.raw_weight, torch.zeros_like(test_model.monotonic.raw_weight))
    assert torch.allclose(test_model.monotonic.bias, torch.zeros_like(test_model.monotonic.bias))


def test_monotonic_nn_init_weights_empty_model():
    """
    Test _init_weights handles edge cases like empty models.
    """
    # Test with no branches at all
    model = MonotonicNN(all_variables=["x1", "x2"])

    # Call initialization - should handle gracefully
    model._init_weights()

    # Verify no layers were created
    assert model.lin_non is None
    assert model.lin_pos is None
    assert model.lin_neg is None
    assert model.out_non is None
    assert model.out_pos is None
    assert model.out_neg is None

    # Test with single variable and single branch
    model = MonotonicNN(all_variables=["x1"], non_monotonic_vars=["x1"], hidden_non=1)

    model._init_weights()

    if model.lin_non:
        assert model.lin_non.weight.shape == (1, 1)
        assert model.lin_non.bias.shape == (1,)
    if model.out_non:
        assert model.out_non.weight.shape == (1, 1)
        assert model.out_non.bias.shape == (1,)

    # Test with zero-width branches that should not create layers
    model = MonotonicNN(
        all_variables=["x1", "x2"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        hidden_non=0,
        hidden_pos=0,
    )

    model._init_weights()
    assert model.lin_non is None
    assert model.out_non is None
    assert model.lin_pos is None
    assert model.out_pos is None


def test_monotonic_nn_init_weights_module_iteration():
    """
    Test that _init_weights correctly iterates through all modules.
    """
    # Create model with nested structure to test module iteration
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3", "x4"],
        non_monotonic_vars=["x1", "x2"],
        positive_monotonic_vars=["x3"],
        negative_monotonic_vars=["x4"],
        hidden_non=2,
        hidden_pos=2,
        hidden_neg=2,
    )

    # Wrap the model in a container to test nested module iteration
    class ContainerModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.extra_linear = nn.Linear(3, 3)

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                if isinstance(m, MonotonicLinear):
                    nn.init.xavier_uniform_(m.raw_weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    container = ContainerModel(model)

    # Verify initial state - weights should be uninitialized (random)
    if container.model.lin_non:
        initial_weight = container.model.lin_non.weight.clone()
    if container.model.lin_pos:
        initial_raw_weight = container.model.lin_pos.raw_weight.clone()

    # Call initialization on container
    container._init_weights()

    # Verify container's own layer was initialized
    assert container.extra_linear.weight is not None
    assert not torch.allclose(container.extra_linear.weight, torch.zeros_like(container.extra_linear.weight))

    # Verify model layers inside container were also initialized
    if container.model.lin_non:
        assert not torch.allclose(container.model.lin_non.weight, initial_weight)
    if container.model.lin_pos:
        assert not torch.allclose(container.model.lin_pos.raw_weight, initial_raw_weight)

    # Test with no modules (edge case)
    class EmptyModel(nn.Module):
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                if isinstance(m, MonotonicLinear):
                    nn.init.xavier_uniform_(m.raw_weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    empty = EmptyModel()
    empty._init_weights()  # Should not raise any errors


def test_monotonic_nn_init_weights_custom_layer_handling():
    """
    Test _init_weights handles custom layers and unexpected types.
    """

    # Create model with custom layers that should be ignored
    class CustomLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 3))
            self.bias = nn.Parameter(torch.randn(3))

    model = MonotonicNN(
        all_variables=["x1", "x2", "x3"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3"],
        hidden_non=2,
        hidden_pos=2,
        hidden_neg=2,
    )

    # Add custom layers that should be ignored by _init_weights
    model.custom_layer1 = CustomLayer()
    model.custom_layer2 = nn.Sequential(nn.Linear(4, 4), CustomLayer())

    # Call initialization
    model._init_weights()

    # Verify custom layers were NOT modified (except the Linear inside sequential)
    initial_custom_weight = model.custom_layer1.weight.clone()
    model._init_weights()  # Call again
    assert torch.allclose(model.custom_layer1.weight, initial_custom_weight)

    # Verify the Linear layer inside sequential WAS modified
    initial_seq_weight = model.custom_layer2[0].weight.clone()
    model._init_weights()
    assert not torch.allclose(model.custom_layer2[0].weight, initial_seq_weight)

    # Test with a layer that has weight but no bias attribute
    class PartialLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3, 3))

    model.partial_layer = PartialLayer()
    model._init_weights()

    # Verify standard layers still work
    if model.lin_non:
        assert model.lin_non.weight is not None
        assert model.lin_non.bias is not None

    # Test with a layer that has bias but no weight (edge case)
    class BiasOnlyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.randn(3))

    model.bias_only = BiasOnlyLayer()
    model._init_weights()  # Should not crash


def test_monotonic_nn_forward_empty_branches():
    """
    Test MonotonicNN.forward when some or all branches are empty.
    """
    # Test with only non-monotonic branch
    model = MonotonicNN(all_variables=["x1", "x2", "x3"], non_monotonic_vars=["x1", "x2"], hidden_non=2)

    x = torch.randn(3, 3)
    output = model.forward(x)
    assert output.shape == (3, 1)
    assert torch.all(torch.isfinite(output))

    # Test with only positive monotonic branch
    model = MonotonicNN(all_variables=["x1", "x2", "x3"], positive_monotonic_vars=["x1", "x2"], hidden_pos=2)

    x = torch.randn(3, 3)
    output = model.forward(x)
    assert output.shape == (3, 1)
    assert torch.all(torch.isfinite(output))

    # Test with only negative monotonic branch
    model = MonotonicNN(all_variables=["x1", "x2", "x3"], negative_monotonic_vars=["x1", "x2"], hidden_neg=2)

    x = torch.randn(3, 3)
    output = model.forward(x)
    assert output.shape == (3, 1)
    assert torch.all(torch.isfinite(output))

    # Test with no branches at all
    model = MonotonicNN(all_variables=["x1", "x2", "x3"])

    x = torch.randn(3, 3)
    output = model.forward(x)
    # Should return zeros since no branches contribute
    assert torch.allclose(output, torch.zeros(3, 1), atol=1e-5)

    # Test with some branches None and others present
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3", "x4"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        hidden_non=0,  # Should make lin_non None
        hidden_pos=2,
    )

    x = torch.randn(3, 4)
    output = model.forward(x)
    assert output.shape == (3, 1)
    assert torch.all(torch.isfinite(output))


def test_monotonic_nn_forward_edge_cases():
    """
    Test MonotonicNN.forward with edge case inputs and model configurations.
    """
    # Test with single variable
    model = MonotonicNN(all_variables=["x1"], non_monotonic_vars=["x1"], hidden_non=1)

    x = torch.randn(3, 1)
    output = model.forward(x)
    assert output.shape == (3, 1)
    assert torch.all(torch.isfinite(output))

    # Test with very large batch size
    model = MonotonicNN(
        all_variables=["x1", "x2"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        hidden_non=4,
        hidden_pos=3,
    )

    x = torch.randn(10000, 2)  # Large batch
    output = model.forward(x)
    assert output.shape == (10000, 1)
    assert torch.all(torch.isfinite(output))

    # Test with very small values
    x_small = torch.tensor([[1e-6, 2e-6], [3e-6, 4e-6]])
    output = model.forward(x_small)
    assert torch.all(torch.isfinite(output))

    # Test with very large values
    x_large = torch.tensor([[1e6, 2e6], [3e6, 4e6]])
    output = model.forward(x_large)
    assert torch.all(torch.isfinite(output))

    # Test with mixed positive/negative inputs
    x_mixed = torch.tensor([[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0]])
    model = MonotonicNN(
        all_variables=["x1", "x2"],
        positive_monotonic_vars=["x1"],
        negative_monotonic_vars=["x2"],
        hidden_pos=2,
        hidden_neg=2,
    )

    output = model.forward(x_mixed)
    assert output.shape == (3, 1)
    assert torch.all(torch.isfinite(output))

    # Test with all zero inputs
    x_zero = torch.zeros(3, 2)
    model = MonotonicNN(all_variables=["x1", "x2"], non_monotonic_vars=["x1", "x2"], hidden_non=1)

    output = model.forward(x_zero)
    assert output.shape == (3, 1)
    assert torch.all(torch.isfinite(output))


def test_monotonic_nn_fit_device_handling():
    """
    Test MonotonicNN.fit with different device configurations.
    """
    # Create model
    model = MonotonicNN(
        all_variables=["x1", "x2"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        hidden_non=4,
        hidden_pos=3,
    )

    # Generate synthetic data
    x_tr = torch.randn(100, 2)
    y_tr = (x_tr[:, 0] + x_tr[:, 1] > 0).float()

    # Test CPU training
    history_cpu = model.fit(x_tr=x_tr, y_tr=y_tr, epochs=1, device="cpu", verbose=False)

    assert history_cpu["train_loss"][0] >= 0

    # Test CUDA training if available
    if torch.cuda.is_available():
        model.cuda()
        x_tr_cuda = x_tr.cuda()
        y_tr_cuda = y_tr.cuda()

        history_cuda = model.fit(x_tr=x_tr_cuda, y_tr=y_tr_cuda, epochs=1, device="cuda", verbose=False)

        assert history_cuda["train_loss"][0] >= 0

        # Test model moves to device
        assert next(model.parameters()).device.type == "cuda"

    # Test invalid device
    with pytest.raises(RuntimeError):
        model.fit(x_tr=x_tr, y_tr=y_tr, epochs=1, device="invalid_device", verbose=False)


def test_monotonic_nn_fit_with_pos_weight():
    """
    Test MonotonicNN.fit with different positive class weights.
    """
    # Create model
    model = MonotonicNN(
        all_variables=["x1", "x2"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        hidden_non=4,
        hidden_pos=3,
    )

    # Generate imbalanced data (more negative class)
    x_tr = torch.randn(200, 2)
    y_tr = (x_tr[:, 0] + x_tr[:, 1] > 0.5).float()  # Biased toward 0
    x_val = torch.randn(50, 2)
    y_val = (x_val[:, 0] + x_val[:, 1] > 0.5).float()

    # Test with default pos_weight (1.0)
    history_default = model.fit(x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, epochs=2, pos_weight=1.0, verbose=False)

    # Test with higher pos_weight (should focus more on positive class)
    history_high_weight = model.fit(
        x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, epochs=2, pos_weight=10.0, verbose=False
    )

    # Test with lower pos_weight (should focus less on positive class)
    history_low_weight = model.fit(
        x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, epochs=2, pos_weight=0.1, verbose=False
    )

    # The training losses should be different for different pos_weights
    assert history_default["train_loss"] != history_high_weight["train_loss"]
    assert history_default["train_loss"] != history_low_weight["train_loss"]

    # Test with extreme pos_weight values
    # Very high weight
    history_extreme = model.fit(
        x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, epochs=2, pos_weight=1000.0, verbose=False
    )

    # Very low weight
    history_extreme_low = model.fit(
        x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, epochs=2, pos_weight=0.001, verbose=False
    )

    # Test that training completes successfully even with extreme weights
    assert len(history_extreme["train_loss"]) == 2
    assert len(history_extreme_low["train_loss"]) == 2


def test_monotonic_nn_predict_proba_device_handling():
    """
    Test MonotonicNN.predict_proba with different device configurations.
    """
    # Create model
    model = MonotonicNN(
        all_variables=["x1", "x2"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        hidden_non=4,
        hidden_pos=3,
    )

    # Test on CPU
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    probs_cpu = model.predict_proba(x_np)
    assert isinstance(probs_cpu, np.ndarray)

    # Test on CUDA if available
    if torch.cuda.is_available():
        model.cuda()
        # Model should move to GPU automatically
        probs_gpu = model.predict_proba(x_np)
        assert isinstance(probs_gpu, np.ndarray)
        assert not np.allclose(probs_cpu, probs_gpu)  # Should be different due to different weights

        # Test that model is on GPU
        assert next(model.parameters()).device.type == "cuda"

    # Test with model on GPU but input on CPU
    if torch.cuda.is_available():
        model.cuda()
        x_np_cpu = np.array([[1.0, 2.0], [3.0, 4.0]])
        probs = model.predict_proba(x_np_cpu)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (2,)


def test_monotonic_nn_predict_logits_device_handling():
    """
    Test MonotonicNN.predict_logits with different device configurations.
    """
    # Create model
    model = MonotonicNN(
        all_variables=["x1", "x2"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        hidden_non=4,
        hidden_pos=3,
    )

    # Test on CPU
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    logits_cpu = model.predict_logits(x_np)
    assert isinstance(logits_cpu, torch.Tensor)

    # Test on CUDA if available
    if torch.cuda.is_available():
        model.cuda()
        # Model should move to GPU automatically
        logits_gpu = model.predict_logits(x_np)
        assert isinstance(logits_gpu, torch.Tensor)
        assert logits_gpu.device.type == "cuda"

        # Test that model is on GPU
        assert next(model.parameters()).device.type == "cuda"

    # Test with model on GPU but input on CPU
    if torch.cuda.is_available():
        model.cuda()
        x_np_cpu = np.array([[1.0, 2.0], [3.0, 4.0]])
        logits = model.predict_logits(x_np_cpu)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 1)


def test_monotonic_nn_predict_logits_with_initialized_model():
    """
    Test MonotonicNN.predict_logits with pre-initialized weights.
    """
    # Create model
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3", "x4"],
        non_monotonic_vars=["x1", "x2"],
        positive_monotonic_vars=["x3"],
        negative_monotonic_vars=["x4"],
        hidden_non=2,
        hidden_pos=2,
        hidden_neg=2,
    )

    # Initialize all weights manually
    if model.lin_non:
        init.xavier_uniform_(model.lin_non.weight)
        init.zeros_(model.lin_non.bias)
    if model.out_non:
        init.xavier_uniform_(model.out_non.weight)
        init.zeros_(model.out_non.bias)

    if model.lin_pos:
        init.xavier_uniform_(model.lin_pos.raw_weight)
        init.zeros_(model.lin_pos.bias)
    if model.out_pos:
        init.xavier_uniform_(model.out_pos.raw_weight)
        init.zeros_(model.out_pos.bias)

    if model.lin_neg:
        init.xavier_uniform_(model.lin_neg.raw_weight)
        init.zeros_(model.lin_neg.bias)
    if model.out_neg:
        init.xavier_uniform_(model.out_neg.raw_weight)
        init.zeros_(model.out_neg.bias)

    # Test forward pass works with initialized weights
    x_np = np.random.randn(5, 4)
    logits = model.predict_logits(x_np)
    assert logits.shape == (5, 1)
    assert isinstance(logits, torch.Tensor)
    assert torch.all(torch.isfinite(logits))

    # Test multiple predictions produce consistent results
    logits1 = model.predict_logits(x_np)
    logits2 = model.predict_logits(x_np)
    assert torch.allclose(logits1, logits2, atol=1e-5)

    # Test that model is in eval mode after prediction
    assert model.training == False


def test_monotonic_nn_permutation_importance_device_handling():
    """
    Test MonotonicNN.permutation_importance with different device configurations.
    """
    # Create model
    model = MonotonicNN(
        all_variables=["x1", "x2"],
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        hidden_non=4,
        hidden_pos=3,
    )

    # Create synthetic data
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 1, 0, 1])

    # Test on CPU
    importances_cpu = model.permutation_importance(X, y, n_repeats=2, device="cpu", seed=42)
    assert isinstance(importances_cpu, np.ndarray)

    # Test on CUDA if available
    if torch.cuda.is_available():
        importances_gpu = model.permutation_importance(X, y, n_repeats=2, device="cuda", seed=42)
        assert isinstance(importances_gpu, np.ndarray)
        assert not np.allclose(importances_cpu, importances_gpu)  # Should be different due to different weights

        # Test that model is on GPU
        assert next(model.parameters()).device.type == "cuda"

    # Test invalid device
    with pytest.raises(RuntimeError):
        model.permutation_importance(X, y, n_repeats=2, device="invalid_device", seed=42)


def test_monotonic_nn_permutation_importance_shape_and_batch_handling():
    """
    Test MonotonicNN.permutation_importance with various input shapes and batch sizes.
    """
    # Test single sample
    model = MonotonicNN(all_variables=["x1", "x2"])
    X_single = np.array([[1.0, 2.0]])
    y_single = np.array([1])

    with pytest.raises(ValueError):
        model.permutation_importance(
            X_single, y_single, n_repeats=2, seed=42
        )  # Need more samples for meaningful importance

    # Test multiple samples
    X_multiple = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_multiple = np.array([0, 1, 0, 1])

    importances = model.permutation_importance(X_multiple, y_multiple, n_repeats=2, seed=42)
    assert importances.shape == (2,)
    assert importances.dtype == np.float64

    # Test very large batch size
    X_large = np.random.randn(1000, 2)
    y_large = np.random.randint(0, 2, (1000,))

    importances_large = model.permutation_importance(X_large, y_large, n_repeats=2, seed=42)
    assert importances_large.shape == (2,)
    assert importances_large.dtype == np.float64

    # Test with single feature
    model_single_feature = MonotonicNN(all_variables=["x1"])
    X_single_feature = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_single_feature = np.array([0, 1, 0, 1])

    importances_single = model_single_feature.permutation_importance(
        X_single_feature, y_single_feature, n_repeats=2, seed=42
    )
    assert importances_single.shape == (1,)

    # Test with very small values
    X_small = np.array([[1e-6, 2e-6], [3e-6, 4e-6], [5e-6, 6e-6], [7e-6, 8e-6]])
    y_small = np.array([0, 1, 0, 1])

    importances_small = model.permutation_importance(X_small, y_small, n_repeats=2, seed=42)
    assert importances_small.shape == (2,)
    assert np.all(np.isfinite(importances_small))

    # Test with very large values
    X_large = np.array([[1e6, 2e6], [3e6, 4e6], [5e6, 6e6], [7e6, 8e6]])
    y_large = np.array([0, 1, 0, 1])

    importances_large = model.permutation_importance(X_large, y_large, n_repeats=2, seed=42)
    assert importances_large.shape == (2,)
    assert np.all(np.isfinite(importances_large))


def test_monotonic_nn_permutation_importance_with_initialized_model():
    """
    Test MonotonicNN.permutation_importance with pre-initialized weights.
    """
    # Create model
    model = MonotonicNN(
        all_variables=["x1", "x2", "x3", "x4"],
        non_monotonic_vars=["x1", "x2"],
        positive_monotonic_vars=["x3"],
        negative_monotonic_vars=["x4"],
        hidden_non=2,
        hidden_pos=2,
        hidden_neg=2,
    )

    # Initialize all weights manually
    if model.lin_non:
        init.xavier_uniform_(model.lin_non.weight)
        init.zeros_(model.lin_non.bias)
    if model.out_non:
        init.xavier_uniform_(model.out_non.weight)
        init.zeros_(model.out_non.bias)

    if model.lin_pos:
        init.xavier_uniform_(model.lin_pos.raw_weight)
        init.zeros_(model.lin_pos.bias)
    if model.out_pos:
        init.xavier_uniform_(model.out_pos.raw_weight)
        init.zeros_(model.out_pos.bias)

    if model.lin_neg:
        init.xavier_uniform_(model.lin_neg.raw_weight)
        init.zeros_(model.lin_neg.bias)
    if model.out_neg:
        init.xavier_uniform_(model.out_neg.raw_weight)
        init.zeros_(model.out_neg.bias)

    # Test with initialized model
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, (100,))

    importances = model.permutation_importance(X, y, n_repeats=2, seed=42)
    assert importances.shape == (4,)
    assert importances.dtype == np.float64
    assert np.all(np.isfinite(importances))

    # Test multiple calls produce consistent results (same seed)
    importances1 = model.permutation_importance(X, y, n_repeats=2, seed=42)
    importances2 = model.permutation_importance(X, y, n_repeats=2, seed=42)
    assert np.allclose(importances1, importances2, atol=1e-5)

    # Test that model is in eval mode after permutation importance
    assert model.training == False


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
