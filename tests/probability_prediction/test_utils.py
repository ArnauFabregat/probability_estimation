import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix

from probability_prediction.utils import (
    calculate_brier_metrics,
    deterministic_baseline,
    get_best_f1,
    stochastic_baseline,
)


def test_stochastic_baseline_default_parameters():
    """
    Test stochastic_baseline with default parameters (N=100, p=0.3, seed=42).
    """
    # Act
    cm, precision, recall, f1 = stochastic_baseline()

    # Assert
    # Verify confusion matrix shape
    assert cm.shape == (2, 2), "Confusion matrix should be 2x2"

    # Verify metrics are floats
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(f1, float), "F1-score should be a float"

    # Verify metrics are in valid range [0, 1]
    assert 0.0 <= precision <= 1.0, "Precision should be between 0 and 1"
    assert 0.0 <= recall <= 1.0, "Recall should be between 0 and 1"
    assert 0.0 <= f1 <= 1.0, "F1-score should be between 0 and 1"


def test_stochastic_baseline_reproducibility():
    """
    Test that stochastic_baseline produces reproducible results with the same seed.
    """
    # Run with same seed twice
    cm1, precision1, recall1, f1_1 = stochastic_baseline(N=100, p=0.3, seed=42)
    cm2, precision2, recall2, f1_2 = stochastic_baseline(N=100, p=0.3, seed=42)

    # Verify results are identical
    assert np.array_equal(cm1, cm2), "Confusion matrices should be identical with same seed"
    assert precision1 == precision2, "Precision should be identical with same seed"
    assert recall1 == recall2, "Recall should be identical with same seed"
    assert f1_1 == f1_2, "F1-score should be identical with same seed"

    # Run with different seed
    cm3, precision3, recall3, f1_3 = stochastic_baseline(N=100, p=0.3, seed=43)

    # Verify results are different (high probability)
    assert not np.array_equal(cm1, cm3), "Confusion matrices should differ with different seeds"
    assert not np.allclose([precision1, recall1, f1_1], [precision3, recall3, f1_3]), (
        "Metrics should differ with different seeds"
    )


def test_deterministic_baseline_default_parameters():
    """
    Test deterministic_baseline with default parameters (N=100, p=0.3).
    """
    # Act
    cm, precision, recall, f1 = deterministic_baseline()

    # Assert
    # Verify confusion matrix shape
    assert cm.shape == (2, 2), "Confusion matrix should be 2x2"

    # Verify metrics are floats
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(f1, float), "F1-score should be a float"

    # Verify metrics are in valid range [0, 1]
    assert 0.0 <= precision <= 1.0, "Precision should be between 0 and 1"
    assert 0.0 <= recall <= 1.0, "Recall should be between 0 and 1"
    assert 0.0 <= f1 <= 1.0, "F1-score should be between 0 and 1"

    # Verify predictions are all ones (deterministic behavior)
    y_true = np.array([1] * 30 + [0] * 70)  # p=0.3, N=100
    y_pred = np.ones(100)
    cm_expected = confusion_matrix(y_true, y_pred)
    assert np.array_equal(cm, cm_expected), "Confusion matrix should match expected deterministic output"


def test_deterministic_baseline_reproducibility():
    """
    Test that deterministic_baseline produces identical results every time.
    """
    # Run twice with same parameters
    cm1, precision1, recall1, f1_1 = deterministic_baseline(N=100, p=0.3)
    cm2, precision2, recall2, f1_2 = deterministic_baseline(N=100, p=0.3)

    # Verify results are identical (deterministic behavior)
    assert np.array_equal(cm1, cm2), "Confusion matrices should be identical (deterministic)"
    assert precision1 == precision2, "Precision should be identical (deterministic)"
    assert recall1 == recall2, "Recall should be identical (deterministic)"
    assert f1_1 == f1_2, "F1-score should be identical (deterministic)"

    # Verify predictions are always ones
    y_true1 = np.array([1] * 30 + [0] * 70)  # p=0.3, N=100
    y_pred1 = np.ones(100)
    cm_expected = confusion_matrix(y_true1, y_pred1)
    assert np.array_equal(cm1, cm_expected), "Confusion matrix should match expected deterministic output"


def test_get_best_f1_basic_functionality():
    """
    Test get_best_f1 with basic binary classification data.
    """
    # Create test data: perfect classifier (y_proba == y_true)
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.95, 0.3, 0.85, 0.4])

    # Act
    best_threshold, best_f1, f1_curve, thresholds = get_best_f1(y_true, y_proba)

    # Assert
    # Verify return types
    assert isinstance(best_threshold, float), "Best threshold should be a float"
    assert isinstance(best_f1, float), "Best F1 should be a float"
    assert isinstance(f1_curve, np.ndarray), "F1 curve should be a numpy array"
    assert isinstance(thresholds, np.ndarray), "Thresholds should be a numpy array"

    # Verify array lengths match
    assert len(f1_curve) == len(thresholds) == 200, "Arrays should have num_thresholds elements"

    # Verify best F1 is achievable
    assert 0.0 <= best_f1 <= 1.0, "Best F1 should be between 0 and 1"

    # Verify best threshold is within [0, 1]
    assert 0.0 <= best_threshold <= 1.0, "Best threshold should be between 0 and 1"

    # Verify f1_curve contains valid F1 scores
    assert np.all((f1_curve >= 0.0) & (f1_curve <= 1.0)), "F1 curve should contain valid F1 scores"


def test_get_best_f1_random_classifier():
    """
    Test get_best_f1 with a random classifier (y_proba are random).
    """
    # Random classifier: y_proba are random values
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_proba = np.random.rand(len(y_true))

    # Act
    best_threshold, best_f1, f1_curve, thresholds = get_best_f1(y_true, y_proba)

    # Assert
    # Verify best F1 is between 0 and 1
    assert 0.0 <= best_f1 <= 1.0, "Best F1 should be between 0 and 1 for random classifier"

    # Verify best threshold is within [0, 1]
    assert 0.0 <= best_threshold <= 1.0, "Best threshold should be between 0 and 1"

    # Verify f1_curve contains valid F1 scores
    assert np.all((f1_curve >= 0.0) & (f1_curve <= 1.0)), "F1 curve should contain valid F1 scores"

    # Verify the curve is not constant (should vary with threshold)
    assert not np.allclose(f1_curve, f1_curve[0]), "F1 curve should vary with threshold for random classifier"

    # Verify the curve has the expected length
    assert len(f1_curve) == 200, "F1 curve should have 200 elements by default"
    assert len(thresholds) == 200, "Thresholds should have 200 elements by default"

    # Verify thresholds are evenly spaced
    assert np.allclose(np.diff(thresholds), (1.0 / 199)), "Thresholds should be evenly spaced between 0 and 1"


def test_calculate_brier_metrics_basic_functionality():
    """
    Test calculate_brier_metrics with basic binary classification data.
    """
    # Create test data: perfect classifier (y_proba == y_true)
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    # Act
    bs_model, bs_baseline, bss, prevalence = calculate_brier_metrics(y_true, y_proba)

    # Assert
    # Verify return types
    assert isinstance(bs_model, float), "Brier Score should be a float"
    assert isinstance(bs_baseline, float), "Baseline Brier Score should be a float"
    assert isinstance(bss, float), "Brier Skill Score should be a float"
    assert isinstance(prevalence, float), "Prevalence should be a float"

    # Verify prevalence calculation
    expected_prevalence = np.mean(y_true)
    assert prevalence == expected_prevalence, "Prevalence should match the mean of y_true"

    # Verify perfect classifier: model BS should be 0
    assert bs_model == 0.0, "Perfect classifier should have Brier Score of 0"

    # Verify baseline BS calculation
    baseline_probs = np.full_like(y_proba, fill_value=expected_prevalence)
    expected_bs_baseline = brier_score_loss(y_true, baseline_probs)
    assert bs_baseline == expected_bs_baseline, "Baseline Brier Score should match manual calculation"

    # Verify BSS calculation
    if bs_baseline == 0:
        assert bss == 0.0, "BSS should be 0 when baseline BS is 0"
    else:
        expected_bss = 1.0 - (bs_model / bs_baseline)
        assert bss == expected_bss, "BSS should match manual calculation"

    # Verify BSS for perfect classifier should be 1.0
    assert bss == 1.0, "Perfect classifier should have BSS of 1.0"


def test_calculate_brier_metrics_imbalanced_data():
    """
    Test calculate_brier_metrics with imbalanced class distributions.
    """
    # Highly imbalanced data (few positive samples)
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 1 positive out of 10
    y_proba = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.07, 0.01, 0.04, 0.06, 0.9])

    # Act
    bs_model, bs_baseline, bss, prevalence = calculate_brier_metrics(y_true, y_proba)

    # Assert
    # Verify prevalence calculation
    expected_prevalence = np.mean(y_true)
    assert prevalence == expected_prevalence, "Prevalence should match the mean of y_true"
    assert prevalence == 0.1, "Prevalence should be 0.1 for this imbalanced dataset"

    # Verify baseline calculation
    baseline_probs = np.full_like(y_proba, fill_value=expected_prevalence)
    expected_bs_baseline = brier_score_loss(y_true, baseline_probs)
    assert bs_baseline == expected_bs_baseline, "Baseline Brier Score should match manual calculation"

    # Verify model performance
    assert bs_model >= 0.0, "Model Brier Score should be non-negative"

    # Verify BSS calculation
    if bs_baseline == 0:
        assert bss == 0.0, "BSS should be 0 when baseline BS is 0"
    else:
        expected_bss = 1.0 - (bs_model / bs_baseline)
        assert bss == expected_bss, "BSS should match manual calculation"

    # Test with very imbalanced data (one positive in large dataset)
    y_true = np.zeros(1000)
    y_true[0] = 1  # Only one positive sample
    y_proba = np.random.rand(1000)

    bs_model, bs_baseline, bss, prevalence = calculate_brier_metrics(y_true, y_proba)
    assert prevalence == 0.001, "Prevalence should be 0.001 for this highly imbalanced dataset"

    # Test with no positive samples
    y_true = np.zeros(100)
    y_proba = np.random.rand(100)

    bs_model, bs_baseline, bss, prevalence = calculate_brier_metrics(y_true, y_proba)
    assert prevalence == 0.0, "Prevalence should be 0 when there are no positive samples"

    # Test with no negative samples
    y_true = np.ones(100)
    y_proba = np.random.rand(100)

    bs_model, bs_baseline, bss, prevalence = calculate_brier_metrics(y_true, y_proba)
    assert prevalence == 1.0, "Prevalence should be 1 when there are no negative samples"
