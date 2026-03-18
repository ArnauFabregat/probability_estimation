from unittest.mock import Mock, patch

import numpy as np
import pytest

from probability_prediction.utils import (
    calculate_brier_metrics,
    deterministic_baseline,
    get_best_f1,
    ice_pdp_plot,
    stochastic_baseline,
)


def test_stochastic_baseline_default_params():
    """
    Test stochastic_baseline with default parameters.
    """
    np.random.seed(42)

    # Run with default parameters
    cm, precision, recall, f1 = stochastic_baseline(verbose=False)

    # Check return types
    assert isinstance(cm, np.ndarray)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)

    # Check confusion matrix shape
    assert cm.shape == (2, 2)

    # Check that confusion matrix sums to N (default N=100)
    assert cm.sum() == 100

    # Check metrics are in valid range [0, 1]
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

    # Check that metrics are finite
    assert np.isfinite(precision)
    assert np.isfinite(recall)
    assert np.isfinite(f1)


def test_stochastic_baseline_custom_params():
    """
    Test stochastic_baseline with custom parameters.
    """
    np.random.seed(42)

    # Run with custom parameters
    N = 200
    p = 0.7
    seed = 123
    cm, precision, recall, f1 = stochastic_baseline(N=N, p=p, seed=seed, verbose=False)

    # Check confusion matrix shape
    assert cm.shape == (2, 2)

    # Check that confusion matrix sums to N
    assert cm.sum() == N

    # Check metrics are in valid range
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

    # Check that metrics are finite
    assert np.isfinite(precision)
    assert np.isfinite(recall)
    assert np.isfinite(f1)

    # Check that the function is reproducible with same seed
    cm2, precision2, recall2, f1_2 = stochastic_baseline(N=N, p=p, seed=seed, verbose=False)
    assert np.array_equal(cm, cm2)
    assert precision == precision2
    assert recall == recall2
    assert f1 == f1_2


def test_stochastic_baseline_verbose_output(capsys):
    """
    Test stochastic_baseline with verbose=True captures output.
    """
    # Run with verbose=True
    cm, precision, recall, f1 = stochastic_baseline(N=50, p=0.5, seed=42, verbose=True)

    # Capture printed output
    captured = capsys.readouterr()

    # Check that output contains expected strings
    assert "=== STOCHASTIC BASELINE ===" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "Precision=" in captured.out
    assert "Recall=" in captured.out
    assert "F1=" in captured.out

    # Check that the output contains the actual values
    assert str(precision) in captured.out or f"{precision:.3f}" in captured.out
    assert str(recall) in captured.out or f"{recall:.3f}" in captured.out
    assert str(f1) in captured.out or f"{f1:.3f}" in captured.out

    # Check that the function still returns correct values
    assert cm.shape == (2, 2)
    assert cm.sum() == 50
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1


def test_deterministic_baseline_default_params():
    """
    Test deterministic_baseline with default parameters.
    """
    # Run with default parameters (N=100, p=0.3)
    cm, precision, recall, f1 = deterministic_baseline(verbose=False)

    # Check return types
    assert isinstance(cm, np.ndarray)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)

    # Check confusion matrix shape
    assert cm.shape == (2, 2)

    # Check that confusion matrix sums to N (default N=100)
    assert cm.sum() == 100

    # With all predictions as class 1, recall should be 1.0
    assert recall == 1.0

    # Precision should be p (30/100 = 0.3)
    assert abs(precision - 0.3) < 0.01

    # Check metrics are in valid range [0, 1]
    assert 0 <= precision <= 1
    assert 0 <= f1 <= 1

    # Check that metrics are finite
    assert np.isfinite(precision)
    assert np.isfinite(recall)
    assert np.isfinite(f1)


def test_deterministic_baseline_custom_params():
    """
    Test deterministic_baseline with custom parameters.
    """
    # Run with custom parameters
    N = 200
    p = 0.5
    cm, precision, recall, f1 = deterministic_baseline(N=N, p=p, verbose=False)

    # Check confusion matrix shape
    assert cm.shape == (2, 2)

    # Check that confusion matrix sums to N
    assert cm.sum() == N

    # With all predictions as class 1, recall should be 1.0
    assert recall == 1.0

    # Precision should be p (0.5 for balanced)
    assert abs(precision - 0.5) < 0.01

    # Check metrics are in valid range
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

    # Check that metrics are finite
    assert np.isfinite(precision)
    assert np.isfinite(recall)
    assert np.isfinite(f1)


def test_deterministic_baseline_verbose_output(capsys):
    """
    Test deterministic_baseline with verbose=True captures output.
    """
    # Run with verbose=True
    cm, precision, recall, f1 = deterministic_baseline(N=50, p=0.4, verbose=True)

    # Capture printed output
    captured = capsys.readouterr()

    # Check that output contains expected strings
    assert "DETERMINISTIC BASELINE" in captured.out
    assert "Confusion matrix:" in captured.out
    assert "Precision=" in captured.out
    assert "Recall=" in captured.out
    assert "F1=" in captured.out

    # Check that the function still returns correct values
    assert cm.shape == (2, 2)
    assert cm.sum() == 50

    # With all predictions as class 1, recall should be 1.0
    assert recall == 1.0

    # Precision should be p (0.4)
    assert abs(precision - 0.4) < 0.02

    # Check metrics are in valid range
    assert 0 <= precision <= 1
    assert 0 <= f1 <= 1


def test_get_best_f1_basic_functionality():
    """
    Test get_best_f1 with typical binary classification probabilities.
    """
    # Create ground truth labels
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])

    # Create predicted probabilities (some good predictions)
    y_proba = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15, 0.75, 0.25])

    # Run function
    best_threshold, best_f1, f1_curve, thresholds = get_best_f1(y_true, y_proba)

    # Check return types
    assert isinstance(best_threshold, float)
    assert isinstance(best_f1, float)
    assert isinstance(f1_curve, np.ndarray)
    assert isinstance(thresholds, np.ndarray)

    # Check that best_threshold is in valid range
    assert 0.0 <= best_threshold <= 1.0

    # Check that best_f1 is in valid range
    assert 0.0 <= best_f1 <= 1.0

    # Check shapes
    assert len(f1_curve) == 200  # default num_thresholds
    assert len(thresholds) == 200

    # Check that thresholds are in order
    assert np.all(np.diff(thresholds) >= 0)

    # Check that best_f1 is actually the max of f1_curve
    assert best_f1 == np.max(f1_curve)


def test_get_best_f1_perfect_predictions():
    """
    Test get_best_f1 with perfect predictions (F1 should be 1.0).
    """
    # Create ground truth labels
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])

    # Create perfect predicted probabilities
    y_proba = np.array([0.99, 0.95, 0.98, 0.01, 0.05, 0.02, 0.97, 0.03])

    # Run function
    best_threshold, best_f1, f1_curve, thresholds = get_best_f1(y_true, y_proba)

    # With perfect predictions, best F1 should be 1.0
    assert abs(best_f1 - 1.0) < 0.01

    # Best threshold should separate the classes well
    assert 0.05 < best_threshold < 0.95

    # Check that f1_curve contains valid values
    assert np.all(np.isfinite(f1_curve))
    assert np.all((f1_curve >= 0) & (f1_curve <= 1))


def test_get_best_f1_custom_thresholds():
    """
    Test get_best_f1 with custom num_thresholds parameter.
    """
    # Create ground truth labels
    y_true = np.array([1, 0, 1, 0, 1])

    # Create predicted probabilities
    y_proba = np.array([0.8, 0.3, 0.6, 0.4, 0.7])

    # Test with different num_thresholds
    num_thresholds = 50
    best_threshold, best_f1, f1_curve, thresholds = get_best_f1(y_true, y_proba, num_thresholds=num_thresholds)

    # Check that output arrays have correct length
    assert len(f1_curve) == num_thresholds
    assert len(thresholds) == num_thresholds

    # Check that thresholds span [0, 1]
    assert thresholds[0] == 0.0
    assert thresholds[-1] == 1.0

    # Test with larger num_thresholds
    num_thresholds_large = 500
    best_threshold2, best_f1_2, f1_curve2, thresholds2 = get_best_f1(
        y_true, y_proba, num_thresholds=num_thresholds_large
    )

    assert len(f1_curve2) == num_thresholds_large
    assert len(thresholds2) == num_thresholds_large

    # Best F1 should be similar (same data, just more thresholds)
    assert abs(best_f1 - best_f1_2) < 0.01


def test_get_best_f1_edge_cases():
    """
    Test get_best_f1 with edge cases.
    """
    # Test with all same true labels (all class 1)
    y_true_all_ones = np.array([1, 1, 1, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    best_threshold, best_f1, f1_curve, thresholds = get_best_f1(y_true_all_ones, y_proba)

    # With all class 1, recall is always 1, precision varies
    # Best F1 should be achieved at lowest threshold
    assert 0.0 <= best_f1 <= 1.0
    assert np.all(np.isfinite(f1_curve))

    # Test with all same true labels (all class 0)
    y_true_all_zeros = np.array([0, 0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    best_threshold2, best_f1_2, f1_curve2, thresholds2 = get_best_f1(y_true_all_zeros, y_proba)

    # With all class 0 and high thresholds, no positive predictions
    # F1 should be 0 or very low
    assert 0.0 <= best_f1_2 <= 1.0
    assert np.all(np.isfinite(f1_curve2))

    # Test with list inputs (should work with array-like)
    y_true_list = [1, 0, 1, 0]
    y_proba_list = [0.8, 0.2, 0.7, 0.3]

    best_threshold3, best_f1_3, f1_curve3, thresholds3 = get_best_f1(y_true_list, y_proba_list)

    assert isinstance(best_threshold3, float)
    assert isinstance(best_f1_3, float)
    assert 0.0 <= best_f1_3 <= 1.0


def test_calculate_brier_metrics_basic():
    """
    Test calculate_brier_metrics with typical binary classification data.
    """
    # Create ground truth labels
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])

    # Create predicted probabilities (reasonable predictions)
    y_proba = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15])

    # Run function
    bs_model, bs_baseline, bss, prevalence = calculate_brier_metrics(y_true, y_proba)

    # Check return types
    assert isinstance(bs_model, float)
    assert isinstance(bs_baseline, float)
    assert isinstance(bss, float)
    assert isinstance(prevalence, float)

    # Check prevalence is correct (4/8 = 0.5)
    assert abs(prevalence - 0.5) < 0.01

    # Check Brier scores are non-negative
    assert bs_model >= 0
    assert bs_baseline >= 0

    # Check Brier scores are finite
    assert np.isfinite(bs_model)
    assert np.isfinite(bs_baseline)
    assert np.isfinite(bss)

    # With reasonable predictions, model should outperform baseline
    assert bss > 0


def test_calculate_brier_metrics_perfect_predictions():
    """
    Test calculate_brier_metrics with perfect predictions.
    BSS should be 1.0 for a perfect model.
    """
    # Create ground truth labels
    y_true = np.array([1, 1, 0, 0, 1, 0])

    # Create perfect predicted probabilities
    y_proba = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

    # Run function
    bs_model, bs_baseline, bss, prevalence = calculate_brier_metrics(y_true, y_proba)

    # Perfect predictions should have Brier Score of 0
    assert abs(bs_model - 0.0) < 0.001

    # BSS should be 1.0 for perfect model
    assert abs(bss - 1.0) < 0.01

    # Check prevalence (3/6 = 0.5)
    assert abs(prevalence - 0.5) < 0.01

    # Baseline Brier Score should be positive
    assert bs_baseline > 0


def test_calculate_brier_metrics_baseline_zero():
    """
    Test calculate_brier_metrics when baseline Brier Score is zero.
    This happens when all true labels are the same.
    """
    # All same true labels (all class 1)
    y_true = np.array([1, 1, 1, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    # Run function
    bs_model, bs_baseline, bss, prevalence = calculate_brier_metrics(y_true, y_proba)

    # Prevalence should be 1.0
    assert abs(prevalence - 1.0) < 0.01

    # Baseline Brier Score should be 0 (predicting 1.0 for all class 1)
    assert abs(bs_baseline - 0.0) < 0.001

    # When bs_baseline is 0, BSS should be 0 (as per function logic)
    assert abs(bss - 0.0) < 0.001

    # Model Brier Score should be positive (imperfect predictions)
    assert bs_model > 0

    # Test with all class 0
    y_true_zeros = np.array([0, 0, 0, 0, 0])
    y_proba_zeros = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    bs_model2, bs_baseline2, bss2, prevalence2 = calculate_brier_metrics(y_true_zeros, y_proba_zeros)

    # Prevalence should be 0.0
    assert abs(prevalence2 - 0.0) < 0.01

    # Baseline Brier Score should be 0
    assert abs(bs_baseline2 - 0.0) < 0.001

    # BSS should be 0
    assert abs(bss2 - 0.0) < 0.001


def test_calculate_brier_metrics_worse_than_baseline():
    """
    Test calculate_brier_metrics when model underperforms baseline.
    BSS should be negative.
    """
    # Create ground truth labels
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    # Create intentionally bad predictions (inverted)
    y_proba_bad = np.array([0.1, 0.2, 0.1, 0.15, 0.9, 0.85, 0.95, 0.8])

    # Run function
    bs_model, bs_baseline, bss, prevalence = calculate_brier_metrics(y_true, y_proba_bad)

    # Prevalence should be 0.5 (4/8)
    assert abs(prevalence - 0.5) < 0.01

    # Model Brier Score should be higher than baseline (worse predictions)
    assert bs_model > bs_baseline

    # BSS should be negative (model underperforms baseline)
    assert bss < 0

    # Check that all values are finite
    assert np.isfinite(bs_model)
    assert np.isfinite(bs_baseline)
    assert np.isfinite(bss)

    # Test with list inputs (array-like compatibility)
    y_true_list = [1, 0, 1, 0]
    y_proba_list = [0.3, 0.7, 0.4, 0.6]  # Somewhat inverted

    bs_model2, bs_baseline2, bss2, prevalence2 = calculate_brier_metrics(y_true_list, y_proba_list)

    assert isinstance(bs_model2, float)
    assert isinstance(bss2, float)
    assert 0.0 <= prevalence2 <= 1.0


def test_ice_pdp_plot_feature_not_found():
    np.random.seed(42)
    mock_model = Mock()
    mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
    X_std = np.random.randn(5, 3)
    X_raw = np.random.randn(5, 3)
    all_vars = ["feature_a", "feature_b", "feature_c"]
    with patch("probability_prediction.utils.plt"):
        with pytest.raises(ValueError, match="not in all_vars"):
            ice_pdp_plot(
                model=mock_model,
                X_std=X_std,
                X_raw=X_raw,
                feature_name="non_existent_feature",
                all_vars=all_vars,
                num_points=10,
            )
