import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.optim as optim


# ============================
# 1. PLATT CALIBRATION
# ============================

class PlattCalibrator:
    """
    Classic Platt scaling: logistic regression on raw probabilities.
    Works for any model with predict_proba().
    """

    def __init__(self):
        self.lr = LogisticRegression()

    def fit(self, probs, y):
        probs = np.asarray(probs).reshape(-1, 1)
        y = np.asarray(y)
        self.lr.fit(probs, y)

    def predict_proba(self, probs):
        probs = np.asarray(probs).reshape(-1, 1)
        return self.lr.predict_proba(probs)[:, 1]


# ============================
# 2. ISOTONIC REGRESSION
# ============================

class IsotonicCalibrator:
    """
    Non-parametric monotonic mapping.
    Best when you have large validation data (> 10k).
    """

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs, y):
        probs = np.asarray(probs)
        y = np.asarray(y)
        self.iso.fit(probs, y)

    def predict_proba(self, probs):
        probs = np.asarray(probs)
        return self.iso.predict(probs)


# ============================
# 3. TEMPERATURE SCALING (PyTorch)
# ============================

class TemperatureScaler(nn.Module):
    """
    Temperature scaling for neural networks.
    Model must output logits (before sigmoid).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, X_tensor, y_tensor, max_iter=50):
        self.model.eval()
        logits = self.model(X_tensor).detach()

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled_logits, y_tensor.float()
            )
            loss.backward()
            return loss

        optimizer.step(closure)

    def predict_proba(self, X_tensor):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            scaled = logits / self.temperature
            return torch.sigmoid(scaled).cpu().numpy().reshape(-1)


# ============================
# 4. UNIFIED CALIBRATOR WRAPPER
# ============================

class Calibrator:
    """
    Unified calibrator for:
    - XGBoost
    - Any sklearn-like model with predict_proba()
    - PyTorch models with logits (for temperature scaling)

    method: "platt", "isotonic", "temperature"
    """

    def __init__(self, method="platt", model=None):
        self.method = method
        self.model = model

        if method == "platt":
            self.cal = PlattCalibrator()
        elif method == "isotonic":
            self.cal = IsotonicCalibrator()
        elif method == "temperature":
            assert model is not None, "Temperature scaling requires the base model."
            self.cal = TemperatureScaler(model)
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
