import numpy as np
from sklearn.linear_model import LogisticRegression


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
