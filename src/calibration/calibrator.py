from typing import Literal, Union, Any
from numpy.typing import ArrayLike, NDArray
import numpy as np

from src.calibration.platt_scaling import PlattCalibrator
from src.calibration.isotonic_regression import IsotonicCalibrator
from src.calibration.temperature_scaling import TemperatureScaler


class Calibrator:
    """
    Unified probability calibrator supporting:

    - **Platt scaling**
    - **Isotonic regression**
    - **Temperature scaling** (for neural networks / logits)

    Parameters
    ----------
    method : {"platt", "isotonic", "temperature"}, default="platt"
        Calibration technique to use.

    Attributes
    ----------
    method : str
        Selected calibration method.

    cal : object
        Underlying calibrator instance:
        - `PlattCalibrator`
        - `IsotonicCalibrator`
        - `TemperatureScaler`

    Notes
    -----
    - For *Platt* and *Isotonic*: inputs must be **probabilities** `p(y=1)`.
    - For *Temperature scaling*: inputs must be **logits** from a neural network.
    """

    def __init__(self, method: Literal["platt", "isotonic", "temperature"] = "platt") -> None:
        self.method = method

        if method == "platt":
            self.cal = PlattCalibrator()
        elif method == "isotonic":
            self.cal = IsotonicCalibrator()  # type: ignore
        elif method == "temperature":
            self.cal = TemperatureScaler()  # type: ignore
        else:
            raise ValueError("Unknown method: choose 'platt', 'isotonic', or 'temperature'")

    def fit(
        self,
        X_val: Union[ArrayLike, Any],
        y_val: ArrayLike,
    ) -> None:
        """
        Fit the calibration model.

        Parameters
        ----------
        X_val :
            - For *Platt* and *Isotonic*: predicted probabilities.
              Shape can be (n_samples,) or (n_samples, 2).
            - For *Temperature scaling*: logits tensor (PyTorch).
        y_val : array-like of shape (n_samples,)
            Ground-truth binary labels.

        Raises
        ------
        ValueError
            If the method is unknown or inputs have invalid shape.
        """
        if self.method in ("platt", "isotonic"):
            X_val_arr = np.asarray(X_val)

            if X_val_arr.ndim > 1:
                raw_probs: NDArray[np.float64] = X_val_arr[:, 1].astype(float)
            else:
                raw_probs = X_val_arr.astype(float)

            self.cal.fit(raw_probs, y_val)

        elif self.method == "temperature":
            # Pass raw logits to TemperatureScaler
            self.cal.fit(X_val, y_val)

    def predict_proba(
        self,
        X: Union[ArrayLike, Any]
    ) -> NDArray[np.float64]:
        """
        Predict calibrated probabilities.

        Parameters
        ----------
        X :
            - For *Platt* and *Isotonic*: uncalibrated probabilities.
            Shape (n_samples,) or (n_samples, 2).
            - For *Temperature scaling*: logits tensor.

        Returns
        -------
        calibrated : ndarray of shape (n_samples,)
            Calibrated probabilities `p(y=1)`.

        Notes
        -----
        The returned array is always converted to `float64`.
        """
        if self.method in ("platt", "isotonic"):
            X_arr = np.asarray(X)

            if X_arr.ndim > 1:
                raw_probs: NDArray[np.float64] = X_arr[:, 1].astype(float)
            else:
                raw_probs = X_arr.astype(float)

            calibrated = self.cal.predict_proba(raw_probs)
            return np.asarray(calibrated, dtype=float)

        else:
            calibrated = self.cal.predict_proba(X)
            return np.asarray(calibrated, dtype=float)
