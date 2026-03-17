from probability_prediction.calibration.isotonic_regression import IsotonicCalibrator


def test_isotonic_calibrator_init_basic():
    """
    Test IsotonicCalibrator.__init__ with default parameters.
    """
    # Test with default parameters
    calibrator = IsotonicCalibrator()
    assert calibrator.iso is not None
    assert calibrator.iso.y_min is None
    assert calibrator.iso.y_max is None
    assert calibrator.iso.increasing is True
    assert calibrator.iso.out_of_bounds == "clip"

    # Test with custom parameters
    calibrator = IsotonicCalibrator(y_min=0.1, y_max=0.9, increasing=False, out_of_bounds="raise")
    assert calibrator.iso.y_min == 0.1
    assert calibrator.iso.y_max == 0.9
    assert calibrator.iso.increasing is False
    assert calibrator.iso.out_of_bounds == "raise"

    # Test with None bounds
    calibrator = IsotonicCalibrator(y_min=None, y_max=None)
    assert calibrator.iso.y_min is None
    assert calibrator.iso.y_max is None
    assert calibrator.iso.increasing is True


def test_isotonic_calibrator_init_edge_cases():
    """
    Test IsotonicCalibrator.__init__ with edge case inputs.
    """
    # Test with y_min > y_max (should work, but may produce empty range)
    calibrator = IsotonicCalibrator(y_min=0.9, y_max=0.1)
    assert calibrator.iso.y_min == 0.9
    assert calibrator.iso.y_max == 0.1

    # Test with extreme values
    calibrator = IsotonicCalibrator(y_min=-1.0, y_max=2.0)
    assert calibrator.iso.y_min == -1.0
    assert calibrator.iso.y_max == 2.0

    # Test with very small values
    calibrator = IsotonicCalibrator(y_min=1e-10, y_max=1e-5)
    assert calibrator.iso.y_min == 1e-10
    assert calibrator.iso.y_max == 1e-5

    # Test with None bounds (should use defaults)
    calibrator = IsotonicCalibrator(y_min=None, y_max=None)
    assert calibrator.iso.y_min is None
    assert calibrator.iso.y_max is None

    # Test with large values
    calibrator = IsotonicCalibrator(y_min=1e6, y_max=1e9)
    assert calibrator.iso.y_min == 1e6
    assert calibrator.iso.y_max == 1e9

    # Test with negative increasing
    calibrator = IsotonicCalibrator(increasing=False)
    assert calibrator.iso.increasing is False
