import numpy as np
import torch

from probability_prediction.calibration.temperature_scaling import TemperatureScaler


def test_temperature_scaler_closure_with_optimizer_integration():
    """
    Test TemperatureScaler.closure with optimizer integration.
    """
    # Create test data
    logits = torch.tensor([0.5, 1.0, -0.5, 2.0, -1.0])
    y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)

    # Initialize calibrator
    calibrator = TemperatureScaler()

    # Set up optimizer
    optimizer = torch.optim.LBFGS([calibrator.log_T], lr=0.01, max_iter=1)

    # Define closure function
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = logits / calibrator.temperature
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, y)
        loss.backward()  # type: ignore
        return loss

    # Act - run optimizer step
    optimizer.step(closure)

    # Assert
    # Verify optimizer state
    assert hasattr(optimizer, "state"), "Optimizer should have state"
    assert len(optimizer.state) > 0, "Optimizer state should not be empty"

    # Verify temperature changed
    initial_temperature = 1.0  # Default initial temperature
    assert not np.isclose(calibrator.temperature.item(), initial_temperature, rtol=1e-5, atol=1e-5), (
        "Temperature should have changed after optimization"
    )

    # Verify temperature is positive
    assert calibrator.temperature.item() > 0, "Temperature should be positive"

    # Verify temperature is not zero
    assert not np.isclose(calibrator.temperature.item(), 0.0), "Temperature should not be zero"

    # Verify gradients were updated
    assert calibrator.log_T.grad is not None, "log_T should have gradients after optimization"
    assert calibrator.log_T.grad.shape == calibrator.log_T.shape, "Gradient shape should match log_T"

    # Verify gradients are not zero
    assert not torch.allclose(calibrator.log_T.grad, torch.zeros_like(calibrator.log_T.grad)), (
        "Gradients should not be all zeros"
    )

    # Verify temperature is learnable
    assert calibrator.log_T.requires_grad is True, "log_T should be learnable"

    # Test with multiple optimizer steps
    for _ in range(3):
        optimizer.step(closure)

    # Verify temperature continues to change
    assert not np.isclose(calibrator.temperature.item(), initial_temperature, rtol=1e-5, atol=1e-5), (
        "Temperature should continue to change"
    )

    # Verify temperature is positive
    assert calibrator.temperature.item() > 0, "Temperature should remain positive"

    # Verify temperature is not zero
    assert not np.isclose(calibrator.temperature.item(), 0.0), "Temperature should not be zero"

    # Verify gradients are still computed
    assert calibrator.log_T.grad is not None, "log_T should still have gradients"
    assert calibrator.log_T.grad.shape == calibrator.log_T.shape, "Gradient shape should match log_T"

    # Verify loss computation works
    scaled = logits / calibrator.temperature
    loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, y)
    assert torch.isfinite(loss).item(), "Loss should be finite"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Verify closure can be called after optimization
    final_loss = closure()
    assert final_loss is not None, "Final closure call should return loss"
    assert torch.isfinite(final_loss).item(), "Final loss should be finite"
    assert final_loss.item() >= 0, "Final loss should be non-negative"

    # Verify temperature remains positive
    assert calibrator.temperature.item() > 0, "Temperature should remain positive"

    # Verify temperature is not zero
    assert not np.isclose(calibrator.temperature.item(), 0.0), "Temperature should not be zero"
