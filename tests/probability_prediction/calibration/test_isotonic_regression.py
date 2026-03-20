import numpy as np

from probability_prediction.calibration.isotonic_regression import IsotonicCalibrator


def test_isotonic_calibrator_predict_proba_basic_functionality():
    """
    Test IsotonicCalibrator.predict_proba with basic functionality.
    """
    # Create test data
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Initialize and fit calibrator
    calibrator = IsotonicCalibrator()
    calibrator.fit(probs, y)

    # Act
    calibrated = calibrator.predict_proba(probs)

    # Assert
    # Verify return type
    assert isinstance(calibrated, np.ndarray), "Should return numpy array"
    assert calibrated.dtype == np.float64, "Should return float64 array"

    # Verify shape
    assert calibrated.shape == (10,), "Should return array with 10 elements"

    # Verify values are in valid range
    assert np.all((calibrated >= 0) & (calibrated <= 1)), "Calibrated probabilities should be in [0, 1]"

    # Verify monotonic relationship (increasing by default)
    assert np.all(np.diff(calibrated) >= 0), "Calibrated probabilities should be monotonic increasing"

    # Verify transformed values differ from original (calibration effect)
    assert not np.allclose(calibrated, probs, rtol=1e-2, atol=1e-2), "Transformed values should differ from original"

    # Verify transformed values are not identical to input
    assert not np.array_equal(calibrated, probs), "Transformed values should not be identical to input"

    # Test with single value
    single_calibrated = calibrator.predict_proba(0.5)
    assert isinstance(single_calibrated, np.ndarray), "Should return numpy array"
    assert single_calibrated.shape == (1,), "Should return array with 1 element"
    assert single_calibrated.dtype == np.float64, "Should return float64 array"

    # Test with list input
    list_calibrated = calibrator.predict_proba([0.1, 0.5, 0.9])
    assert isinstance(list_calibrated, np.ndarray), "Should return numpy array"
    assert list_calibrated.shape == (3,), "Should return array with 3 elements"

    # Test with different probabilities
    new_probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    new_calibrated = calibrator.predict_proba(new_probs)
    assert new_calibrated.shape == (5,), "Should return array with 5 elements"
    assert np.all((new_calibrated >= 0) & (new_calibrated <= 1)), "New calibrated probabilities should be in [0, 1]"
