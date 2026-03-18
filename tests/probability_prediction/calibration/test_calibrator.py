from unittest.mock import Mock

import numpy as np
import pytest
import torch

from probability_prediction.calibration.calibrator import Calibrator
from probability_prediction.calibration.isotonic_regression import IsotonicCalibrator
from probability_prediction.calibration.platt_scaling import PlattCalibrator
from probability_prediction.calibration.temperature_scaling import TemperatureScaler


def test_calibrator_init_platt():
    """
    Test Calibrator initialization with platt method.
    """
    calibrator = Calibrator(method="platt")

    assert calibrator.method == "platt"
    assert hasattr(calibrator, "cal")
    assert isinstance(calibrator.cal, PlattCalibrator)


def test_calibrator_init_isotonic():
    """
    Test Calibrator initialization with isotonic method.
    """
    calibrator = Calibrator(method="isotonic")

    assert calibrator.method == "isotonic"
    assert hasattr(calibrator, "cal")
    assert isinstance(calibrator.cal, IsotonicCalibrator)


def test_calibrator_init_temperature():
    """
    Test Calibrator initialization with temperature method.
    """
    calibrator = Calibrator(method="temperature")

    assert calibrator.method == "temperature"
    assert hasattr(calibrator, "cal")
    assert isinstance(calibrator.cal, TemperatureScaler)


def test_calibrator_init_invalid_method():
    """
    Test Calibrator initialization with invalid method raises ValueError.
    """
    with pytest.raises(ValueError, match="Unknown method"):
        Calibrator(method="invalid_method")

    with pytest.raises(ValueError, match="Unknown method"):
        Calibrator(method="")

    with pytest.raises(ValueError, match="Unknown method"):
        Calibrator(method="softmax")


def test_calibrator_fit_platt_1d():
    """
    Test Calibrator.fit with platt method and 1D probabilities.
    """
    calibrator = Calibrator(method="platt")

    # Mock the underlying calibrator's fit method
    calibrator.cal.fit = Mock()

    # Create test data
    X_val = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y_val = np.array([0, 0, 1, 1, 1])

    calibrator.fit(X_val, y_val)

    # Verify the underlying calibrator's fit was called with correct args
    calibrator.cal.fit.assert_called_once()
    call_args = calibrator.cal.fit.call_args
    np.testing.assert_array_equal(call_args[0][0], X_val)
    np.testing.assert_array_equal(call_args[0][1], y_val)


def test_calibrator_fit_platt_2d():
    """
    Test Calibrator.fit with platt method and 2D probabilities (extract column 1).
    """
    calibrator = Calibrator(method="platt")

    # Mock the underlying calibrator's fit method
    calibrator.cal.fit = Mock()

    # Create 2D test data (e.g., from predict_proba output)
    X_val = np.array([[0.9, 0.1], [0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    y_val = np.array([0, 0, 1, 1, 1])

    calibrator.fit(X_val, y_val)

    # Verify the underlying calibrator's fit was called
    calibrator.cal.fit.assert_called_once()
    call_args = calibrator.cal.fit.call_args

    # Should extract column 1 (probabilities for positive class)
    expected_probs = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
    np.testing.assert_array_equal(call_args[0][0], expected_probs)
    np.testing.assert_array_equal(call_args[0][1], y_val)


def test_calibrator_fit_isotonic():
    """
    Test Calibrator.fit with isotonic method.
    """
    calibrator = Calibrator(method="isotonic")

    # Mock the underlying calibrator's fit method
    calibrator.cal.fit = Mock()

    # Create test data
    X_val = np.array([0.2, 0.4, 0.6, 0.8])
    y_val = np.array([0, 0, 1, 1])

    calibrator.fit(X_val, y_val)

    # Verify the underlying calibrator's fit was called
    calibrator.cal.fit.assert_called_once()
    call_args = calibrator.cal.fit.call_args
    np.testing.assert_array_equal(call_args[0][0], X_val)
    np.testing.assert_array_equal(call_args[0][1], y_val)


def test_calibrator_fit_temperature():
    """
    Test Calibrator.fit with temperature method (passes logits directly).
    """
    calibrator = Calibrator(method="temperature")

    # Mock the underlying calibrator's fit method
    calibrator.cal.fit = Mock()

    # Create test data (logits, not probabilities)
    X_val = torch.tensor([1.0, -0.5, 0.8, -1.2, 0.3])
    y_val = np.array([1, 0, 1, 0, 1])

    calibrator.fit(X_val, y_val)

    # Verify the underlying calibrator's fit was called with logits directly
    calibrator.cal.fit.assert_called_once()
    call_args = calibrator.cal.fit.call_args
    assert torch.equal(call_args[0][0], X_val)
    np.testing.assert_array_equal(call_args[0][1], y_val)
