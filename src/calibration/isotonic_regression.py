import numpy as np
from sklearn.isotonic import IsotonicRegression


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
