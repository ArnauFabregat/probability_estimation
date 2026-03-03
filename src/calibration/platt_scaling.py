import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """
    Platt scaling for probabilistic calibration.

    This method fits a logistic regression model on top of **uncalibrated
    probabilities or scores** to learn a monotonic, parametric transformation
    that maps them to calibrated probabilities.

    Platt scaling is lightweight and works well with:
    - SVMs (original use-case)
    - tree-based models (XGBoost, LightGBM)
    - any classifier that outputs scores or probabilities

    It is less flexible than isotonic regression but has **lower variance** and
    works well even with small validation sets.

    Parameters
    ----------
    penalty : str, default="l2"
        Regularization penalty used by logistic regression.
    C : float, default=1.0
        Inverse of regularization strength.
    solver : str, default="lbfgs"
        Solver used by logistic regression.
    max_iter : int, default=1000
        Maximum iterations for convergence.

    Attributes
    ----------
    lr : LogisticRegression
        Underlying logistic regression model performing calibration.

    Notes
    -----
    - Inputs are reshaped into column vectors `(n_samples, 1)` because sklearn
      expects 2D feature matrices.
    - The returned probabilities are always 1D arrays of shape `(n_samples,)`,
      representing calibrated `p(y=1)` values.
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
    ) -> None:
        self.lr = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
        )

    def fit(self, probs: ArrayLike, y: ArrayLike) -> None:
        """
        Fit Platt scaling using logistic regression.

        Parameters
        ----------
        probs : array-like of shape (n_samples,)
            Uncalibrated predicted probabilities or scores.
        y : array-like of shape (n_samples,)
            Ground-truth binary labels (0/1).

        Raises
        ------
        ValueError
            If `probs` and `y` do not have matching lengths.
        """
        probs_arr = np.asarray(probs, dtype=float).reshape(-1, 1)
        y_arr = np.asarray(y).reshape(-1)

        if probs_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"`probs` and `y` must have the same length. Got {probs_arr.shape[0]} vs {y_arr.shape[0]}."
            )

        self.lr.fit(probs_arr, y_arr)

    def predict_proba(self, probs: ArrayLike) -> NDArray[np.float64]:
        """
        Predict calibrated probabilities using the fitted Platt model.

        Parameters
        ----------
        probs : array-like of shape (n_samples,)
            Uncalibrated probabilities or scores to calibrate.

        Returns
        -------
        calibrated : ndarray of shape (n_samples,)
            Calibrated probabilities for the positive class (`p(y=1)`).

        Notes
        -----
        Values are passed through logistic regression and clipped to `[0, 1]`
        automatically by the model's output.
        """
        probs_arr = np.asarray(probs, dtype=float).reshape(-1, 1)
        calibrated = self.lr.predict_proba(probs_arr)[:, 1]
        return np.asarray(calibrated, dtype=float)
