import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """
    Non-parametric monotonic probability calibration via isotonic regression.

    This calibrator learns a monotonic mapping from scores/probabilities to
    calibrated probabilities using `sklearn.isotonic.IsotonicRegression` with
    `out_of_bounds="clip"`.

    Isotonic regression is flexible (non-parametric) and can capture complex
    monotonic relationships, but it tends to need **larger** validation sets
    (e.g., > 10k samples) to avoid overfitting.

    Parameters
    ----------
    y_min : float, optional
        Lower bound of the calibrated output. If None, inferred by estimator.
    y_max : float, optional
        Upper bound of the calibrated output. If None, inferred by estimator.
    increasing : bool, default=True
        Whether the mapping is monotonically increasing.
    out_of_bounds : {"nan", "clip", "raise"}, default="clip"
        Behavior for inputs outside the range seen during fitting.
        - "clip": clip to the min/max fitted x-values
        - "nan": return NaN
        - "raise": raise a ValueError

    Attributes
    ----------
    iso : IsotonicRegression
        The underlying scikit-learn isotonic regressor.

    Notes
    -----
    - Inputs (`probs`/`scores`) must be 1D and aligned with `y` (same length).
    - Calibration is typically performed on **uncalibrated probabilities** `p(y=1)`
      or **monotonic scores** (e.g., logits). The learned mapping is monotonic.
    - If you pass logits instead of probabilities, the calibrated outputs will still
      be constrained to `[y_min, y_max]`, typically `[0, 1]` if left as default.
    """

    def __init__(
        self,
        y_min: float | None = None,
        y_max: float | None = None,
        increasing: bool = True,
        out_of_bounds: str = "clip",
    ) -> None:
        self.iso = IsotonicRegression(
            y_min=y_min,
            y_max=y_max,
            increasing=increasing,
            out_of_bounds=out_of_bounds,
        )

    def fit(self, probs: ArrayLike, y: ArrayLike) -> None:
        """
        Fit the isotonic mapping.

        Parameters
        ----------
        probs : array-like of shape (n_samples,)
            Uncalibrated probabilities or monotonic scores. Must be 1D.
        y : array-like of shape (n_samples,)
            Ground-truth binary labels (0/1) or continuous targets (isotonic supports
            real-valued targets, though for calibration it's typically binary).

        Raises
        ------
        ValueError
            If inputs are not 1D arrays or lengths do not match.
        """
        probs_arr = np.asarray(probs, dtype=float).reshape(-1)
        y_arr = np.asarray(y).reshape(-1)

        if probs_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"`probs` and `y` must have the same length. Got {probs_arr.shape[0]} vs {y_arr.shape[0]}."
            )

        self.iso.fit(probs_arr, y_arr)

    def predict_proba(self, probs: ArrayLike) -> NDArray[np.float64]:
        """
        Predict calibrated probabilities using the fitted isotonic mapping.

        Parameters
        ----------
        probs : array-like of shape (n_samples,)
            Uncalibrated probabilities or scores to calibrate. Must be 1D.

        Returns
        -------
        calibrated : ndarray of shape (n_samples,)
            Calibrated probabilities (float64).

        Raises
        ------
        ValueError
            If `probs` is not 1D.
        """
        probs_arr = np.asarray(probs, dtype=float).reshape(-1)
        calibrated = self.iso.predict(probs_arr)
        return np.asarray(calibrated, dtype=float)
