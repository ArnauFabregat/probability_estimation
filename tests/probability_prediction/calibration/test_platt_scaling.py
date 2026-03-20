import numpy as np
from sklearn.linear_model import LogisticRegression

from probability_prediction.calibration.platt_scaling import PlattCalibrator


def test_platt_calibrator_fit_basic_functionality():
    """
    Test PlattCalibrator.fit with basic binary classification data.
    """
    # Create test data
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Initialize calibrator
    calibrator = PlattCalibrator()

    # Act
    calibrator.fit(probs, y)

    # Assert
    # Verify calibrator was fitted
    assert hasattr(calibrator.lr, "coef_"), "LogisticRegression should have coef_ after fitting"
    assert hasattr(calibrator.lr, "intercept_"), "LogisticRegression should have intercept_ after fitting"
    assert hasattr(calibrator.lr, "classes_"), "LogisticRegression should have classes_ after fitting"

    # Verify coef_ and intercept_ have correct shapes
    assert calibrator.lr.coef_.shape == (1, 1), "coef_ should be shape (1, 1) for binary classification"
    assert calibrator.lr.intercept_.shape == (1,), "intercept_ should be shape (1,)"

    # Verify classes_ contains binary labels
    assert np.array_equal(calibrator.lr.classes_, [0, 1]), "classes_ should contain [0, 1]"

    # Verify coef_ and intercept_ are not zero
    assert not np.allclose(calibrator.lr.coef_, 0), "coef_ should not be all zeros"
    assert not np.allclose(calibrator.lr.intercept_, 0), "intercept_ should not be all zeros"

    # Verify calibrator can be used for prediction (basic check)
    probs_2d = np.asarray(probs).reshape(-1, 1)
    predictions = calibrator.lr.predict(probs_2d)
    assert predictions.shape == (10,), "Predictions should have 10 elements"
    assert np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary"

    # Verify predict_proba returns probabilities
    probas = calibrator.lr.predict_proba(probs_2d)
    assert probas.shape == (10, 2), "predict_proba should return (n_samples, 2)"
    assert np.all((probas >= 0) & (probas <= 1)), "Probabilities should be in [0, 1]"
    assert np.allclose(probas.sum(axis=1), 1), "Probabilities should sum to 1 for each sample"

    # Verify calibrated probabilities are different from original
    calibrated_probs = calibrator.lr.predict_proba(probs_2d)[:, 1]
    assert not np.allclose(calibrated_probs, probs, rtol=1e-2, atol=1e-2), (
        "Calibrated probabilities should differ from original"
    )

    # Verify monotonic relationship
    assert np.all(np.diff(calibrated_probs) >= 0), "Calibrated probabilities should be monotonic increasing"


def test_platt_calibrator_fit_with_real_data():
    """
    Test PlattCalibrator.fit with data from a real model.
    """
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.1 * X[:, 2] + np.random.randn(100) * 0.5 > 0).astype(int)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X, y)

    # Get uncalibrated probabilities
    probs = model.predict_proba(X)[:, 1]

    # Initialize and fit calibrator
    calibrator = PlattCalibrator()
    calibrator.fit(probs, y)

    # Verify calibrator was fitted with real data
    assert calibrator.lr.coef_.shape == (1, 1), "coef_ should have shape (1, 1)"
    assert calibrator.lr.intercept_.shape == (1,), "intercept_ should have shape (1,)"
    assert calibrator.lr.classes_.shape == (2,), "classes_ should have shape (2,)"

    # Verify coef_ and intercept_ are not zero
    assert not np.allclose(calibrator.lr.coef_, 0), "coef_ should not be all zeros"
    assert not np.allclose(calibrator.lr.intercept_, 0), "intercept_ should not be all zeros"

    # Verify classes_ contains binary labels
    assert np.array_equal(calibrator.lr.classes_, [0, 1]), "classes_ should contain [0, 1]"

    # Test with custom parameters
    calibrator = PlattCalibrator(C=0.1, max_iter=500)
    calibrator.fit(probs, y)

    # Verify custom parameters were used
    assert calibrator.lr.C == 0.1, "C should be 0.1"
    assert calibrator.lr.max_iter == 500, "max_iter should be 500"

    # Verify fitting completed
    assert calibrator.lr.coef_.shape == (1, 1), "coef_ should have shape (1, 1)"
    assert calibrator.lr.intercept_.shape == (1,), "intercept_ should have shape (1,)"

    # Test with different solvers
    calibrator = PlattCalibrator(solver="liblinear")
    calibrator.fit(probs, y)

    # Verify solver was used
    assert calibrator.lr.solver == "liblinear", "solver should be 'liblinear'"

    # Verify fitting completed
    assert calibrator.lr.coef_.shape == (1, 1), "coef_ should have shape (1, 1)"
    assert calibrator.lr.intercept_.shape == (1,), "intercept_ should have shape (1,)"

    # Test with very small dataset
    small_probs = np.array([0.1, 0.9])
    small_y = np.array([0, 1])

    calibrator = PlattCalibrator()
    calibrator.fit(small_probs, small_y)

    # Verify fitting with small dataset
    assert calibrator.lr.coef_.shape == (1, 1), "coef_ should have shape (1, 1)"
    assert calibrator.lr.intercept_.shape == (1,), "intercept_ should have shape (1,)"

    # Test with very large dataset
    large_probs = np.random.rand(10000)
    large_y = (np.random.rand(10000) > 0.5).astype(int)

    calibrator = PlattCalibrator()
    calibrator.fit(large_probs, large_y)

    # Verify fitting with large dataset
    assert calibrator.lr.coef_.shape == (1, 1), "coef_ should have shape (1, 1)"
    assert calibrator.lr.intercept_.shape == (1,), "intercept_ should have shape (1,)"

    # Verify lr attribute exists and is correct type
    assert hasattr(calibrator, "lr"), "Should have 'lr' attribute"
    assert isinstance(calibrator.lr, LogisticRegression), "lr should be LogisticRegression instance"
