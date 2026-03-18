import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from probability_prediction.model import MonotonicLinear, MonotonicNN
from probability_prediction.schemas import OptimizerParams


def test_monotonic_linear_init_valid_signs():
    """
    Test MonotonicLinear initialization with valid sign values.
    """
    # Test with default sign
    layer1 = MonotonicLinear(in_features=10, out_features=5)
    assert layer1.in_features == 10
    assert layer1.out_features == 5
    assert layer1.sign == "+"
    assert isinstance(layer1.raw_weight, nn.Parameter)
    assert isinstance(layer1.bias, nn.Parameter)
    assert layer1.raw_weight.shape == (5, 10)
    assert layer1.bias.shape == (5,)

    # Test with explicit positive sign
    layer2 = MonotonicLinear(in_features=8, out_features=3, sign="+")
    assert layer2.sign == "+"

    # Test with negative sign
    layer3 = MonotonicLinear(in_features=4, out_features=2, sign="-")
    assert layer3.sign == "-"


def test_monotonic_linear_init_invalid_sign():
    """
    Test MonotonicLinear initialization with invalid sign raises ValueError.
    """
    # Test with invalid sign
    with pytest.raises(ValueError, match="sign must be '\\+' or '-'."):
        MonotonicLinear(in_features=5, out_features=3, sign="invalid")

    # Test with empty string
    with pytest.raises(ValueError, match="sign must be '\\+' or '-'."):
        MonotonicLinear(in_features=5, out_features=3, sign="")

    # Test with numeric sign
    with pytest.raises(ValueError, match="sign must be '\\+' or '-'."):
        MonotonicLinear(in_features=5, out_features=3, sign=1)


def test_monotonic_linear_init_parameter_shapes():
    """
    Test MonotonicLinear parameter shapes match specified dimensions.
    """
    # Test various input/output dimensions
    test_cases = [
        (1, 1),  # Single feature to single output
        (10, 5),  # Standard case
        (100, 50),  # Larger dimensions
        (3, 1),  # Multiple inputs, single output
    ]

    for in_feat, out_feat in test_cases:
        layer = MonotonicLinear(in_features=in_feat, out_features=out_feat)
        assert layer.raw_weight.shape == (out_feat, in_feat)
        assert layer.bias.shape == (out_feat,)

        # Verify parameters are properly initialized as torch Parameters
        assert layer.raw_weight.requires_grad is True
        assert layer.bias.requires_grad is True


def test_monotonic_linear_init_inheritance():
    """
    Test MonotonicLinear properly inherits from nn.Module.
    """
    layer = MonotonicLinear(in_features=5, out_features=3)

    # Verify inheritance
    assert isinstance(layer, nn.Module)

    # Verify module registration
    assert hasattr(layer, "parameters")
    assert hasattr(layer, "named_parameters")

    # Verify the layer is in training mode by default
    assert layer.training is True

    # Verify the layer can be switched to eval mode
    layer.eval()
    assert layer.training is False


def test_monotonic_linear_forward_positive_sign():
    """
    Test MonotonicLinear forward pass with positive monotonicity.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create layer with positive sign
    layer = MonotonicLinear(in_features=3, out_features=2, sign="+")

    # Create input tensor
    batch_size = 4
    x = torch.randn(batch_size, 3)

    # Forward pass
    output = layer(x)

    # Verify output shape
    assert output.shape == (batch_size, 2)

    # Verify monotonicity: weights should be positive (softplus ensures this)
    weight = F.softplus(layer.raw_weight)
    assert torch.all(weight >= 0)

    # Verify computation matches expected formula
    expected = x @ weight.T + layer.bias
    assert torch.allclose(output, expected)


def test_monotonic_linear_forward_negative_sign():
    """
    Test MonotonicLinear forward pass with negative monotonicity.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create layer with negative sign
    layer = MonotonicLinear(in_features=3, out_features=2, sign="-")

    # Create input tensor
    batch_size = 4
    x = torch.randn(batch_size, 3)

    # Forward pass
    output = layer(x)

    # Verify output shape
    assert output.shape == (batch_size, 2)

    # Verify monotonicity: weights should be negative (softplus ensures positive, then negated)
    weight = -F.softplus(layer.raw_weight)
    assert torch.all(weight <= 0)

    # Verify computation matches expected formula
    expected = x @ weight.T + layer.bias
    assert torch.allclose(output, expected)


def test_monotonic_linear_forward_batch_sizes():
    """
    Test MonotonicLinear forward pass with various batch sizes.
    """
    torch.manual_seed(42)

    # Create layer
    layer = MonotonicLinear(in_features=5, out_features=3)

    # Test different batch sizes
    batch_sizes = [1, 8, 32, 128]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 5)
        output = layer(x)

        # Verify output shape
        assert output.shape == (batch_size, 3)

        # Verify output is finite (no NaN or inf)
        assert torch.all(torch.isfinite(output))


def test_monotonic_linear_forward_gradient_flow():
    """
    Test that gradients flow properly through MonotonicLinear forward pass.
    """
    torch.manual_seed(42)

    # Create layer with requires_grad=True (default for Parameters)
    layer = MonotonicLinear(in_features=4, out_features=2)

    # Create input with requires_grad to test gradient flow
    x = torch.randn(3, 4, requires_grad=True)

    # Forward pass
    output = layer(x)

    # Compute a simple loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Verify gradients exist and are finite
    assert layer.raw_weight.grad is not None
    assert layer.bias.grad is not None
    assert x.grad is not None

    assert torch.all(torch.isfinite(layer.raw_weight.grad))
    assert torch.all(torch.isfinite(layer.bias.grad))
    assert torch.all(torch.isfinite(x.grad))


def test_monotonic_nn_init_all_branches():
    """
    Test MonotonicNN initialization with all three branches.
    """
    # Define variables for each branch
    all_vars = ["x1", "x2", "x3", "x4", "x5"]
    non_monotonic = ["x1", "x2"]
    positive = ["x3", "x4"]
    negative = ["x5"]

    # Create model
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=non_monotonic,
        positive_monotonic_vars=positive,
        negative_monotonic_vars=negative,
        hidden_non=16,
        hidden_pos=8,
        hidden_neg=8,
    )

    # Verify variable lists are stored
    assert model.all_variables == all_vars
    assert model.non_monotonic_vars == non_monotonic
    assert model.positive_monotonic_vars == positive
    assert model.negative_monotonic_vars == negative

    # Verify masks are created as buffers
    assert hasattr(model, "mask_non")
    assert hasattr(model, "mask_pos")
    assert hasattr(model, "mask_neg")
    assert model.mask_non.shape == (2,)
    assert model.mask_pos.shape == (2,)
    assert model.mask_neg.shape == (1,)

    # Verify layers are created
    assert model.lin_non is not None
    assert model.lin_pos is not None
    assert model.lin_neg is not None
    assert isinstance(model.lin_pos, MonotonicLinear)
    assert isinstance(model.lin_neg, MonotonicLinear)
    assert model.lin_pos.sign == "+"
    assert model.lin_neg.sign == "-"

    # Verify output layers
    assert model.out_non is not None
    assert model.out_pos is not None
    assert model.out_neg is not None


def test_monotonic_nn_init_empty_branches():
    """
    Test MonotonicNN initialization with empty variable lists.
    """
    # Define variables
    all_vars = ["x1", "x2", "x3"]

    # Create model with empty branches
    model = MonotonicNN(
        all_variables=all_vars, non_monotonic_vars=[], positive_monotonic_vars=[], negative_monotonic_vars=[]
    )

    # Verify empty lists are stored
    assert model.non_monotonic_vars == []
    assert model.positive_monotonic_vars == []
    assert model.negative_monotonic_vars == []

    # Verify masks are None
    assert model.mask_non is None
    assert model.mask_pos is None
    assert model.mask_neg is None

    # Verify layers are None
    assert model.lin_non is None
    assert model.lin_pos is None
    assert model.lin_neg is None
    assert model.out_non is None
    assert model.out_pos is None
    assert model.out_neg is None


def test_monotonic_nn_init_partial_branches():
    """
    Test MonotonicNN initialization with only some branches.
    """
    # Define variables
    all_vars = ["x1", "x2", "x3", "x4"]
    positive = ["x1", "x2"]

    # Create model with only positive branch
    model = MonotonicNN(
        all_variables=all_vars, non_monotonic_vars=[], positive_monotonic_vars=positive, negative_monotonic_vars=[]
    )

    # Verify masks
    assert model.mask_non is None
    assert model.mask_pos is not None
    assert model.mask_neg is None
    assert model.mask_pos.shape == (2,)

    # Verify layers
    assert model.lin_non is None
    assert model.lin_pos is not None
    assert model.lin_neg is None
    assert model.out_non is None
    assert model.out_pos is not None
    assert model.out_neg is None

    # Verify layer properties
    assert model.lin_pos.in_features == 2
    assert model.lin_pos.out_features == 8  # default hidden_pos
    assert model.lin_pos.sign == "+"


def test_monotonic_nn_init_variable_mapping():
    """
    Test MonotonicNN variable name to index mapping.
    """
    # Define variables in specific order
    all_vars = ["feature_a", "feature_b", "feature_c", "feature_d"]
    non_monotonic = ["feature_b", "feature_d"]
    positive = ["feature_a"]
    negative = ["feature_c"]

    # Create model
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=non_monotonic,
        positive_monotonic_vars=positive,
        negative_monotonic_vars=negative,
    )

    # Verify masks contain correct indices
    # feature_b is index 1, feature_d is index 3
    expected_non_mask = torch.tensor([1, 3], dtype=torch.long)
    assert torch.equal(model.mask_non, expected_non_mask)

    # feature_a is index 0
    expected_pos_mask = torch.tensor([0], dtype=torch.long)
    assert torch.equal(model.mask_pos, expected_pos_mask)

    # feature_c is index 2
    expected_neg_mask = torch.tensor([2], dtype=torch.long)
    assert torch.equal(model.mask_neg, expected_neg_mask)


def test_init_weights_empty_branches():
    """
    Test _init_weights handles empty branches gracefully.
    """
    # Create model with no branches
    all_vars = ["x1", "x2", "x3"]
    model = MonotonicNN(
        all_variables=all_vars, non_monotonic_vars=[], positive_monotonic_vars=[], negative_monotonic_vars=[]
    )

    # All branch layers should be None
    assert model.lin_non is None
    assert model.lin_pos is None
    assert model.lin_neg is None
    assert model.out_non is None
    assert model.out_pos is None
    assert model.out_neg is None

    # _init_weights should not crash with no layers
    # (It's called in __init__, so we just verify the model was created)
    assert isinstance(model, nn.Module)


def test_init_weights_reinitialization():
    """
    Test that _init_weights can be called multiple times to reinitialize.
    """
    # Create model
    all_vars = ["x1", "x2", "x3"]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3"],
    )

    # Store initial weights
    initial_lin_non_weight = model.lin_non.weight.clone()
    initial_lin_pos_raw_weight = model.lin_pos.raw_weight.clone()
    initial_lin_neg_raw_weight = model.lin_neg.raw_weight.clone()

    # Call _init_weights again
    model._init_weights()

    # Weights should be different (reinitialized)
    # Note: There's a tiny chance they could be the same by random chance,
    # but with Xavier uniform initialization, it's extremely unlikely
    assert not torch.allclose(model.lin_non.weight, initial_lin_non_weight)
    assert not torch.allclose(model.lin_pos.raw_weight, initial_lin_pos_raw_weight)
    assert not torch.allclose(model.lin_neg.raw_weight, initial_lin_neg_raw_weight)

    # Biases should still be zeros
    assert torch.all(model.lin_non.bias == 0)
    assert torch.all(model.lin_pos.bias == 0)
    assert torch.all(model.lin_neg.bias == 0)


def test_forward_all_branches():
    """
    Test forward pass with all three branches active.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create model with all branches
    all_vars = ["x1", "x2", "x3", "x4", "x5"]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["x1", "x2"],
        positive_monotonic_vars=["x3", "x4"],
        negative_monotonic_vars=["x5"],
        hidden_non=16,
        hidden_pos=8,
        hidden_neg=8,
    )

    # Create input tensor
    batch_size = 4
    x = torch.randn(batch_size, 5)

    # Forward pass
    output = model(x)

    # Verify output shape
    assert output.shape == (batch_size, 1)

    # Verify output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verify output is finite
    assert torch.all(torch.isfinite(output))


def test_forward_single_branch():
    """
    Test forward pass with only one branch active.
    """
    torch.manual_seed(42)

    # Create model with only positive branch
    all_vars = ["x1", "x2", "x3"]
    model = MonotonicNN(
        all_variables=all_vars, non_monotonic_vars=[], positive_monotonic_vars=["x1", "x2"], negative_monotonic_vars=[]
    )

    # Create input tensor
    batch_size = 3
    x = torch.randn(batch_size, 3)

    # Forward pass
    output = model(x)

    # Verify output shape
    assert output.shape == (batch_size, 1)

    # Verify output is finite
    assert torch.all(torch.isfinite(output))

    # Verify that only positive branch is used
    # We can check that the output is computed only from positive branch
    # by comparing with manual computation
    with torch.no_grad():
        # Manual computation for positive branch
        h = torch.tanh(model.lin_pos(x[:, model.mask_pos]))
        expected = model.out_pos(h)
        assert torch.allclose(output, expected)


def test_forward_empty_branches():
    """
    Test forward pass with no branches (edge case).
    """
    torch.manual_seed(42)

    # Create model with no branches
    all_vars = ["x1", "x2", "x3"]
    model = MonotonicNN(
        all_variables=all_vars, non_monotonic_vars=[], positive_monotonic_vars=[], negative_monotonic_vars=[]
    )

    # Create input tensor
    batch_size = 2
    x = torch.randn(batch_size, 3)

    # Forward pass
    output = model(x)

    # Verify output shape
    assert output.shape == (batch_size, 1)

    # With no branches, output should be zeros (initialized in forward)
    expected = torch.zeros((batch_size, 1), device=x.device)
    assert torch.allclose(output, expected)


def test_forward_gradient_flow():
    """
    Test that gradients flow through forward pass.
    """
    torch.manual_seed(42)

    # Create model with all branches
    all_vars = ["x1", "x2", "x3", "x4"]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3", "x4"],
    )

    # Create input with requires_grad
    x = torch.randn(3, 4, requires_grad=True)

    # Forward pass
    output = model(x)

    # Compute loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Verify gradients exist for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient for {name}"

    # Verify input gradient exists
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_fit_basic_training():
    """
    Test basic training loop without validation.
    """
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create simple dataset
    n_samples = 100
    n_features = 4
    x_tr = torch.randn(n_samples, n_features)
    y_tr = torch.randint(0, 2, (n_samples,)).float()

    # Create model
    all_vars = [f"x{i}" for i in range(n_features)]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["x0", "x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3"],
    )

    # Train model
    history = model.fit(x_tr=x_tr, y_tr=y_tr, epochs=3, verbose=False)

    # Check history structure
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 0  # No validation provided

    # Check that training loss is recorded and is finite
    for loss in history["train_loss"]:
        assert np.isfinite(loss)
        assert loss >= 0  # BCE loss is non-negative


def test_fit_with_validation():
    """
    Test training with validation data.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create datasets
    n_samples = 100
    n_features = 3
    x_tr = torch.randn(n_samples, n_features)
    y_tr = torch.randint(0, 2, (n_samples,)).float()
    x_val = torch.randn(20, n_features)
    y_val = torch.randint(0, 2, (20,)).float()

    # Create model
    all_vars = [f"x{i}" for i in range(n_features)]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["x0"],
        positive_monotonic_vars=["x1"],
        negative_monotonic_vars=["x2"],
    )

    # Train with validation
    history = model.fit(x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, epochs=3, verbose=False)

    # Check history structure
    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3

    # Check that both losses are finite
    for train_loss, val_loss in zip(history["train_loss"], history["val_loss"]):
        assert np.isfinite(train_loss)
        assert np.isfinite(val_loss)
        assert train_loss >= 0
        assert val_loss >= 0


def test_fit_early_stopping():
    """
    Test early stopping functionality.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create datasets
    n_samples = 50
    n_features = 2
    x_tr = torch.randn(n_samples, n_features)
    y_tr = torch.randint(0, 2, (n_samples,)).float()
    x_val = torch.randn(10, n_features)
    y_val = torch.randint(0, 2, (10,)).float()

    # Create model
    all_vars = [f"x{i}" for i in range(n_features)]
    model = MonotonicNN(all_variables=all_vars, non_monotonic_vars=["x0"], positive_monotonic_vars=["x1"])

    # Set up early stopping with very small patience
    optimizer_params = OptimizerParams(
        lr=0.001,
        weight_decay=1e-4,
        batch_size=16,
        patience=1,  # Stop after 1 epoch without improvement
        min_delta=10.0,  # Large delta to ensure no improvement
    )

    # Train with early stopping
    history = model.fit(
        x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, epochs=10, optimizer_params=optimizer_params, verbose=False
    )

    # Should stop early due to patience=1 and min_delta=10.0
    # With random data, unlikely to improve by 10.0 after first epoch
    assert len(history["train_loss"]) < 10  # Should be less than max epochs
    assert len(history["val_loss"]) == len(history["train_loss"])

    # Check that training stopped early
    assert len(history["train_loss"]) <= 3  # Should stop within a few epochs


def test_fit_custom_parameters():
    """
    Test training with custom optimizer parameters.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create dataset
    n_samples = 80
    n_features = 3
    x_tr = torch.randn(n_samples, n_features)
    y_tr = torch.randint(0, 2, (n_samples,)).float()

    # Create model
    all_vars = [f"x{i}" for i in range(n_features)]
    model = MonotonicNN(all_variables=all_vars, non_monotonic_vars=["x0", "x1"], positive_monotonic_vars=["x2"])

    # Custom optimizer parameters
    optimizer_params = OptimizerParams(lr=0.01, weight_decay=0.001, batch_size=32, patience=5, min_delta=0.001)

    # Train with custom parameters
    history = model.fit(
        x_tr=x_tr,
        y_tr=y_tr,
        epochs=2,
        optimizer_params=optimizer_params,
        pos_weight=2.0,  # Custom positive class weight
        shuffle=False,  # No shuffling
        verbose=False,
    )

    # Check that training completed
    assert len(history["train_loss"]) == 2

    # Check that loss values are reasonable
    for loss in history["train_loss"]:
        assert 0 <= loss <= 10  # BCE loss should be in reasonable range

    # Verify model parameters were updated
    # Check that at least some parameters have non-zero gradients
    # (This is a basic check that training occurred)
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    assert has_gradients


def test_predict_logits_basic():
    """
    Test basic predict_logits functionality.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    all_vars = ["x1", "x2", "x3"]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3"],
    )

    # Create numpy input
    batch_size = 5
    n_features = 3
    x_np = np.random.randn(batch_size, n_features).astype(np.float32)

    # Get logits
    logits = model.predict_logits(x_np)

    # Verify output type and shape
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, 1)

    # Verify output is on CPU (default device)
    assert logits.device.type == "cpu"

    # Verify model is in eval mode
    assert model.training is False

    # Verify logits are finite
    assert torch.all(torch.isfinite(logits))


def test_predict_logits_vs_forward():
    """
    Test that predict_logits matches forward output.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    all_vars = ["a", "b", "c", "d"]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["a", "b"],
        positive_monotonic_vars=["c"],
        negative_monotonic_vars=["d"],
    )

    # Create numpy input
    batch_size = 3
    n_features = 4
    x_np = np.random.randn(batch_size, n_features).astype(np.float32)

    # Get logits via predict_logits
    logits_from_method = model.predict_logits(x_np)

    # Get logits via forward (manually converting numpy to tensor)
    x_tensor = torch.from_numpy(x_np).float()
    logits_from_forward = model.forward(x_tensor)

    # They should be identical
    assert torch.allclose(logits_from_method, logits_from_forward)

    # Verify both are on CPU
    assert logits_from_method.device.type == "cpu"
    assert logits_from_forward.device.type == "cpu"


def test_predict_logits_device_handling():
    """
    Test predict_logits with model on different device.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    all_vars = ["x1", "x2"]
    model = MonotonicNN(all_variables=all_vars, non_monotonic_vars=["x1"], positive_monotonic_vars=["x2"])

    # Create numpy input
    batch_size = 2
    n_features = 2
    x_np = np.random.randn(batch_size, n_features).astype(np.float32)

    # Test on CPU (default)
    logits_cpu = model.predict_logits(x_np)
    assert logits_cpu.device.type == "cpu"

    # If CUDA is available, test on GPU
    if torch.cuda.is_available():
        model = model.cuda()
        logits_gpu = model.predict_logits(x_np)
        assert logits_gpu.device.type == "cuda"

        # Verify results are the same (within tolerance)
        assert torch.allclose(logits_cpu, logits_gpu.cpu(), rtol=1e-5, atol=1e-5)

        # Move back to CPU for cleanup
        model = model.cpu()


def test_predict_logits_single_sample():
    """
    Test predict_logits with single sample (edge case).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    all_vars = ["feature1", "feature2", "feature3"]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["feature1"],
        positive_monotonic_vars=["feature2"],
        negative_monotonic_vars=["feature3"],
    )

    # Create single sample numpy input
    x_np = np.random.randn(1, 3).astype(np.float32)

    # Get logits
    logits = model.predict_logits(x_np)

    # Verify shape
    assert logits.shape == (1, 1)

    # Verify output is a tensor
    assert isinstance(logits, torch.Tensor)

    # Verify model is in eval mode
    assert model.training is False

    # Verify logits are finite
    assert torch.all(torch.isfinite(logits))


def test_permutation_importance_basic():
    """
    Test basic permutation importance computation.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    all_vars = ["x1", "x2", "x3"]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3"],
    )

    # Create dataset
    n_samples = 50
    n_features = 3
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)

    # Compute permutation importance
    importances = model.permutation_importance(X=X, y=y, n_repeats=3, seed=42)

    # Verify output shape
    assert importances.shape == (n_features,)

    # Verify output is numpy array
    assert isinstance(importances, np.ndarray)

    # Verify importances are finite
    assert np.all(np.isfinite(importances))

    # Verify model is in eval mode after computation
    assert model.training is False


def test_permutation_importance_reproducibility():
    """
    Test that permutation importance is reproducible with same seed.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    all_vars = ["a", "b", "c", "d"]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["a", "b"],
        positive_monotonic_vars=["c"],
        negative_monotonic_vars=["d"],
    )

    # Create dataset
    n_samples = 100
    n_features = 4
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)

    # Compute permutation importance twice with same seed
    importances1 = model.permutation_importance(X=X, y=y, n_repeats=5, seed=123)

    importances2 = model.permutation_importance(X=X, y=y, n_repeats=5, seed=123)

    # Results should be identical
    assert np.allclose(importances1, importances2)

    # Compute with different seed
    importances3 = model.permutation_importance(X=X, y=y, n_repeats=5, seed=456)

    # Results should be different (but close due to small sample)
    # We just verify they're not exactly equal
    assert not np.array_equal(importances1, importances3)


def test_permutation_importance_n_repeats():
    """
    Test permutation importance with different n_repeats values.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    all_vars = ["x1", "x2"]
    model = MonotonicNN(all_variables=all_vars, non_monotonic_vars=["x1"], positive_monotonic_vars=["x2"])

    # Create dataset
    n_samples = 50
    n_features = 2
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)

    # Test with n_repeats=1
    importances_single = model.permutation_importance(X=X, y=y, n_repeats=1, seed=42)
    assert importances_single.shape == (n_features,)
    assert np.all(np.isfinite(importances_single))

    # Test with n_repeats=10
    importances_multiple = model.permutation_importance(X=X, y=y, n_repeats=10, seed=42)
    assert importances_multiple.shape == (n_features,)
    assert np.all(np.isfinite(importances_multiple))

    # Both should have same shape
    assert importances_single.shape == importances_multiple.shape


def test_permutation_importance_single_feature():
    """
    Test permutation importance with single feature (edge case).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model with single feature
    all_vars = ["feature"]
    model = MonotonicNN(
        all_variables=all_vars, non_monotonic_vars=["feature"], positive_monotonic_vars=[], negative_monotonic_vars=[]
    )

    # Create dataset
    n_samples = 30
    n_features = 1
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)

    # Compute permutation importance
    importances = model.permutation_importance(X=X, y=y, n_repeats=3, seed=42)

    # Verify output shape
    assert importances.shape == (1,)

    # Verify output is finite
    assert np.all(np.isfinite(importances))

    # For single feature, importance should be non-negative
    # (permuting the only feature should always increase loss)
    # Note: This is generally true but not guaranteed, so we just check it's a number
    assert isinstance(importances[0], (np.floating, float))


def test_predict_prob_via_permutation_importance():
    """
    Test _predict_prob functionality indirectly through permutation_importance.
    _predict_prob is a nested function inside permutation_importance.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    all_vars = ["x1", "x2", "x3"]
    model = MonotonicNN(
        all_variables=all_vars,
        non_monotonic_vars=["x1"],
        positive_monotonic_vars=["x2"],
        negative_monotonic_vars=["x3"],
    )

    n_samples = 30
    n_features = 3
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)

    importances = model.permutation_importance(X=X, y=y, n_repeats=3, seed=42)

    assert importances.shape == (n_features,)
    assert np.all(np.isfinite(importances))


def test_predict_prob_device_cpu():
    """
    Test _predict_prob correctly handles CPU device.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    all_vars = ["x1", "x2"]
    model = MonotonicNN(all_variables=all_vars, non_monotonic_vars=["x1"], positive_monotonic_vars=["x2"])

    X = np.random.randn(5, 2).astype(np.float32)
    y = np.random.randint(0, 2, 5).astype(np.float32)

    importances = model.permutation_importance(X=X, y=y, n_repeats=2, device="cpu", seed=42)

    assert importances.shape == (2,)
    assert np.all(np.isfinite(importances))
