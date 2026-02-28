from src.calibration.platt_scaling import PlattCalibrator
from src.calibration.isotonic_regression import IsotonicCalibrator
from src.calibration.temperature_scaling import TemperatureScaler


class Calibrator:
    """
    Unified calibrator for:
    - XGBoost
    - Any sklearn-like model with predict_proba()
    - PyTorch models with logits (for temperature scaling)

    method: "platt", "isotonic", "temperature"
    """

    def __init__(self, method="platt"):
        self.method = method

        if method == "platt":
            self.cal = PlattCalibrator()
        elif method == "isotonic":
            self.cal = IsotonicCalibrator()
        elif method == "temperature":
            self.cal = TemperatureScaler()
        else:
            raise ValueError("Unknown method: choose 'platt', 'isotonic', or 'temperature'")

    def fit(self, X_val, y_val):
        """
        For Platt/Isotonic:
            X_val = predicted probabilities
        For Temperature scaling:
            X_val = torch tensor inputs to NN
        """

        if self.method in ("platt", "isotonic"):
            # X_val must be raw probabilities from the base model
            if X_val.ndim > 1:
                raw_probs = X_val[:, 1]  # extract p(y=1)
            else:
                raw_probs = X_val

            self.cal.fit(raw_probs, y_val)

        elif self.method == "temperature":
            # Temperature scaling requires logits before sigmoid
            self.cal.fit(X_val, y_val)

    def predict_proba(self, X):
        """
        For Platt/Isotonic:
            X = raw probs from base model
        For Temperature scaling:
            X = features tensor to NN
        """
        if self.method in ("platt", "isotonic"):
            if X.ndim > 1:
                raw_probs = X[:, 1]
            else:
                raw_probs = X

            calibrated = self.cal.predict_proba(raw_probs)
            return calibrated

        elif self.method == "temperature":
            return self.cal.predict_proba(X)
