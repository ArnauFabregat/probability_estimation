from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import brier_score_loss, f1_score, confusion_matrix, precision_recall_fscore_support


def stochastic_baseline(
    N: int = 100, p: float = 0.3, seed: int = 42, verbose: bool = True
) -> tuple[np.ndarray, float, float, float]:
    """
    Generate predictions from a stochastic no-skill classifier for a binary classification task.

    Description:
    ------------
    This classifier predicts class 1 independently for each sample with probability q = p,
    where p is the prevalence of class 1 in the dataset. This corresponds to the expected
    performance of a model that knows only the marginal class distribution and predicts
    labels by random sampling.

    Parameters:
    -----------
    N : int
        Number of samples.
    p : float
        Proportion of class 1 in the data (0 < p < 1).
    seed : int
        Random seed to ensure reproducibility.
    verbose : bool
        If True, print the confusion matrix and metrics.

    Returns:
    --------
    cm : np.ndarray
        Confusion matrix: [[TN, FP], [FN, TP]]
    precision : float
        Precision for class 1.
    recall : float
        Recall for class 1.
    f1 : float
        F1-score for class 1.
    """
    np.random.seed(seed)

    # Robust probability handling
    p = float(p)
    p = np.clip(p, 0.0, 1.0)  # avoid floating point drift
    q = p
    probs_true = np.array([p, 1 - p])
    probs_pred = np.array([q, 1 - q])
    probs_true /= probs_true.sum()
    probs_pred /= probs_pred.sum()

    # Ground truth
    y_true = np.random.choice([1, 0], size=N, p=probs_true)

    # Predictions
    y_pred = np.random.choice([1, 0], size=N, p=probs_pred)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    if verbose:
        print("=== STOCHASTIC BASELINE ===")
        print("Confusion matrix:\n", cm)
        print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n")

    return cm, precision, recall, f1


def deterministic_baseline(
    N: int = 100, p: float = 0.3, verbose: bool = True
) -> tuple[np.ndarray, float, float, float]:
    """
    Compute the confusion matrix and F1-score for a deterministic no-skill classifier.

    Description:
    ------------
    This baseline classifier predicts class 1 for all samples. In binary classification,
    this corresponds to the thresholding behavior of a model that outputs the same fixed
    probability score equal to class prevalence p, and applies a threshold t <= p.

    This strategy gives the maximum possible F1-score achievable by a classifier that
    only knows the prevalence of class 1, not the features.

    Parameters:
    -----------
    N : int
        Number of samples.
    p : float
        Proportion of class 1 in the data.
    verbose : bool
        If True, print the confusion matrix and metrics.

    Returns:
    --------
    cm : np.ndarray
        Confusion matrix.
    precision : float
        Precision for class 1.
    recall : float
        Recall for class 1.
    f1 : float
        F1-score for class 1.
    """
    # Compute exact number of samples per class
    pos = int(round(N * p))
    neg = N - pos

    # Ground truth length guaranteed to be N
    y_true = np.array([1] * pos + [0] * neg)

    # Predictions (all class 1)
    y_pred = np.ones(N)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    if verbose:
        print("\n=== DETERMINISTIC BASELINE ===")
        print("Confusion matrix:\n", cm)
        print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    return cm, precision, recall, f1


def get_best_f1(
    y_true: ArrayLike, y_proba: ArrayLike, num_thresholds: int = 200
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Computes the optimal F1 score by sweeping probability thresholds.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0/1)

    y_proba : array-like
        Predicted probabilities from model.predict_proba()

    num_thresholds : int, optional (default=200)
        Number of thresholds to test between 0 and 1.

    Returns
    -------
    best_threshold : float
        Probability threshold that maximizes F1 score

    best_f1 : float
        Maximum F1 achieved

    f1_curve : list of floats
        F1 score at each threshold

    thresholds : list of floats
        Tested thresholds (same length as f1_curve)
    """
    # Normalize inputs
    y_true_arr = np.asarray(y_true).astype(int).reshape(-1)
    y_proba_arr = np.asarray(y_proba, dtype=float).reshape(-1)

    thresholds: np.ndarray = np.linspace(0.0, 1.0, num_thresholds, dtype=float)
    f1_curve: np.ndarray = np.empty(num_thresholds, dtype=float)

    for i in range(num_thresholds):
        t = thresholds[i]
        preds = (y_proba_arr >= t).astype(int)
        f1_curve[i] = float(f1_score(y_true_arr, preds))

    best_idx = int(np.argmax(f1_curve))
    best_threshold: float = float(thresholds[best_idx])
    best_f1: float = float(f1_curve[best_idx])

    return best_threshold, best_f1, f1_curve, thresholds


def calculate_brier_metrics(y_true: ArrayLike, y_proba: ArrayLike) -> tuple[float, float, float, float]:
    """
    Computes Brier-based calibration metrics: model Brier Score, baseline Brier Score,
    Brier Skill Score (BSS), and prevalence.

    The Brier Score (BS) is the mean squared error between predicted probabilities and
    binary outcomes. The baseline BS is computed using a naive model that always predicts
    the prevalence (mean of y_true). The Brier Skill Score compares the model to this
    baseline: BSS = 1 - (BS_model / BS_baseline).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0/1).

    y_proba : array-like of shape (n_samples,)
        Predicted probabilities (values in [0, 1]) for the positive class.

    Returns
    -------
    bs_model : float
        Brier Score of the model (lower is better).

    bs_baseline : float
        Brier Score of the baseline model that predicts the prevalence for every sample.

    bss : float
        Brier Skill Score. Values:
            - > 0 : model outperforms the baseline
            - = 0 : model matches the baseline
            - < 0 : model underperforms the baseline
            - = 1 : perfect model (theoretical upper bound)

    prevalence : float
        The positive-class prevalence in `y_true` (mean of y_true).
    """
    # Normalize inputs to arrays with concrete dtypes
    y_true_arr = np.asarray(y_true).astype(int).reshape(-1)
    y_probs_arr = np.asarray(y_proba, dtype=float).reshape(-1)

    bs_model: float = float(brier_score_loss(y_true_arr, y_probs_arr))
    prevalence: float = float(np.mean(y_true_arr))
    baseline_probs = np.full_like(y_probs_arr, fill_value=prevalence, dtype=float)
    bs_baseline: float = float(brier_score_loss(y_true_arr, baseline_probs))

    if bs_baseline == 0:
        bss: float = 0.0
    else:
        bss = 1.0 - (bs_model / bs_baseline)

    return bs_model, bs_baseline, bss, prevalence


# Function generated vibe coding
def ice_pdp_plot(
    model: object,
    X_std: np.ndarray,
    X_raw: np.ndarray,
    feature_name: str,
    all_vars: Sequence[str],
    num_points: int = 50,
    n_samples: int | None = None,
    mode: Literal["ice", "pdp", "both"] = "both",
    calibrator: object | None = None,
    figsize: tuple[int, int] = (8, 5),
    random_state: int = 42,
    grid_percentiles: tuple[float, float] = (0.5, 99.5),
    model_input_space: Literal["auto", "raw", "standardized"] = "auto",
) -> dict[str, Any]:
    """
    Generate ICE (Individual Conditional Expectation) and PDP (Partial Dependence Plot)
    curves for a given feature, supporting both **raw** and **standardized** model
    input representations and multiple calibrator types.

    Key behaviors:
    --------------
    - The **x-axis grid is always in the feature's original raw units** for interpretability.
    - The model receives either standardized or raw inputs based on `model_input_space`:
        * "standardized" → feed `X_std` (recommended for neural nets)
        * "raw" → feed `X_raw` (trees/boosted-trees)
        * "auto" → if temperature scaling is used, assume standardized; else raw.
    - Calibration support:
        * Temperature scaling → expects logits via `model.predict_logits`.
        * Isotonic/Platt → expects probabilities; supports `.predict`, `.transform`,
          or `.predict_proba`.

    Parameters
    ----------
    model : object
        Classifier exposing either `.predict_proba`, `.predict`, and optionally
        `.predict_logits` when using temperature scaling.

    X_std : np.ndarray, shape (n_samples, n_features)
        Standardized feature matrix.

    X_raw : np.ndarray, shape (n_samples, n_features)
        Raw (original scale) feature matrix.

    feature_name : str
        Name of the target feature.

    all_vars : Sequence[str]
        Names of all feature columns.

    num_points : int
        Number of grid points for ICE/PDP.

    n_samples : int or None
        Number of rows to sample for ICE curves. If None → use all rows.

    mode : {"ice", "pdp", "both"}
        Controls which curves are plotted.

    calibrator : object or None
        Calibration model.

    figsize : (int, int)
        Base figure size.

    random_state : int
        RNG seed used when sampling rows.

    grid_percentiles : (float, float)
        Percentile bounds for grid creation in original units.

    model_input_space : {"auto", "raw", "standardized"}
        Controls how data is fed into the model.

    Returns
    -------
    Dict[str, Any]
        {
            "raw_grid": np.ndarray,
            "std_grid": np.ndarray,
            "ice_values": np.ndarray,
            "pdp": np.ndarray,
            "sample_idx": np.ndarray,
            "used_standardized": bool,
        }
    """

    # ------------------------------------------------------
    # Helper: detect temperature scaling
    # ------------------------------------------------------
    def is_temperature_scaler(cal: object | None) -> bool:
        """Return True if calibrator behaves like a temperature scaler."""
        if cal is None:
            return False
        method = getattr(cal, "method", None)
        if isinstance(method, str) and method.lower() == "temperature":
            return True
        return hasattr(cal, "temperature")

    # Decide model input representation
    if model_input_space == "standardized":
        use_standardized_for_model = True
    elif model_input_space == "raw":
        use_standardized_for_model = False
    else:
        # Automatic heuristic
        use_standardized_for_model = is_temperature_scaler(calibrator)

    # ------------------------------------------------------
    # 1) Find feature index
    # ------------------------------------------------------
    if feature_name not in all_vars:
        raise ValueError(f"{feature_name} not in all_vars.")
    idx = all_vars.index(feature_name)

    # ------------------------------------------------------
    # 2) Build raw & standardized grids
    # ------------------------------------------------------
    raw_col: np.ndarray = X_raw[:, idx]
    low, high = np.percentile(raw_col, list(grid_percentiles))

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Grid percentiles produced non-finite bounds.")

    if low == high:
        eps = 1e-6 if low == 0 else abs(low) * 1e-6
        low, high = low - eps, high + eps

    raw_grid: np.ndarray = np.linspace(low, high, num_points)

    mean = raw_col.mean()
    std = raw_col.std(ddof=0)
    std = std if std != 0 else 1.0
    std_grid: np.ndarray = (raw_grid - mean) / std

    # ------------------------------------------------------
    # 3) Sample rows
    # ------------------------------------------------------
    N = X_std.shape[0]
    if n_samples is None or n_samples >= N:
        sample_idx = np.arange(N)
    else:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(N, size=n_samples, replace=False)

    # ------------------------------------------------------
    # Helpers
    # ------------------------------------------------------
    def _to_1d(a: np.ndarray) -> np.ndarray:
        """Ensure an array is flattened into shape (n,)."""
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[1] == 1:
            return a.ravel()
        return a.reshape(-1)

    def get_raw_probs(X_eval: np.ndarray) -> np.ndarray:
        """
        Return model probabilities P(y=1) for X_eval.
        Supports:
        - predict_proba → binary column 1, else column 0.
        - predict → assumed to return probabilities.
        """
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(X_eval))
            if p.ndim == 2:
                return p[:, 1] if p.shape[1] == 2 else p[:, 0]
            return _to_1d(p)

        if hasattr(model, "predict"):
            return _to_1d(model.predict(X_eval))

        raise AttributeError("Model must implement predict_proba(X) or predict(X) returning probabilities.")

    def _apply_probability_calibrator(prob_1d: np.ndarray) -> np.ndarray:
        """
        Apply a probability calibrator to 1‑D probability input.
        Supports: .predict, .transform, .predict_proba.
        """
        prob_1d = _to_1d(prob_1d)
        cal = calibrator

        if hasattr(cal, "predict"):
            return _to_1d(cal.predict(prob_1d))

        if hasattr(cal, "transform"):
            return _to_1d(cal.transform(prob_1d))

        if hasattr(cal, "predict_proba"):
            out = np.asarray(cal.predict_proba(prob_1d))
            return out[:, 1] if out.ndim == 2 and out.shape[1] == 2 else _to_1d(out)

        raise AttributeError("Calibrator must implement predict, transform, or predict_proba.")

    def get_calibrated(X_eval: np.ndarray) -> np.ndarray:
        """
        Return calibrated probabilities.
        Temperature scaling → requires model.predict_logits.
        Isotonic / Platt → calibrates probabilities.
        """
        if calibrator is None:
            return get_raw_probs(X_eval)

        method = getattr(calibrator, "method", None)

        # Temperature scaling
        if method == "temperature" or hasattr(calibrator, "temperature"):
            if not hasattr(model, "predict_logits"):
                raise AttributeError("Temperature scaling requires model.predict_logits(X).")
            logits = np.asarray(model.predict_logits(X_eval)).reshape(-1)
            return _to_1d(calibrator.predict_proba(logits))  # type: ignore

        # Standard probability calibrator
        p_raw = get_raw_probs(X_eval)
        return _apply_probability_calibrator(p_raw)

    # ------------------------------------------------------
    # 4) Compute calibrated ICE
    # ------------------------------------------------------
    ice_values = np.zeros((len(sample_idx), num_points), dtype=float)

    for i, row_idx in enumerate(sample_idx):
        if use_standardized_for_model:
            base = X_std[row_idx].copy()
            X_eval = np.repeat(base[None, :], num_points, axis=0)
            X_eval[:, idx] = std_grid
        else:
            base = X_raw[row_idx].copy()
            X_eval = np.repeat(base[None, :], num_points, axis=0)
            X_eval[:, idx] = raw_grid

        ice_values[i] = get_calibrated(X_eval)

    pdp = ice_values.mean(axis=0)

    # ------------------------------------------------------
    # 5) Compute raw (uncalibrated) ICE/PDP for left plot
    # ------------------------------------------------------
    ice_raw = np.zeros_like(ice_values)

    for i, row_idx in enumerate(sample_idx):
        if use_standardized_for_model:
            base = X_std[row_idx].copy()
            X_eval = np.repeat(base[None, :], num_points, axis=0)
            X_eval[:, idx] = std_grid
        else:
            base = X_raw[row_idx].copy()
            X_eval = np.repeat(base[None, :], num_points, axis=0)
            X_eval[:, idx] = raw_grid

        ice_raw[i] = get_raw_probs(X_eval)

    pdp_raw = ice_raw.mean(axis=0)

    # ------------------------------------------------------
    # 6) Plot
    # ------------------------------------------------------
    _, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # Left: uncalibrated
    ax = axes[0]

    if mode in ("ice", "both"):
        for i in range(len(sample_idx)):
            ax.plot(raw_grid, ice_raw[i], alpha=0.25, color="gray")

    if mode in ("pdp", "both"):
        ax.plot(raw_grid, pdp_raw, linewidth=3, color="blue", label="PDP (raw)")
        ax.legend()

    ax.set_title(f"Uncalibrated — {feature_name}")
    ax.set_xlabel(f"{feature_name} (raw)")
    ax.set_ylabel("Predicted probability")
    ax.grid(True)

    # Right: calibrated
    ax = axes[1]

    if mode in ("ice", "both"):
        for i in range(len(sample_idx)):
            ax.plot(raw_grid, ice_values[i], alpha=0.25, color="gray")

    if mode in ("pdp", "both"):
        ax.plot(raw_grid, pdp, linewidth=3, color="red", label="PDP (calibrated)")
        ax.legend()

    ax.set_title(f"Calibrated — {feature_name}")
    ax.set_xlabel(f"{feature_name} (raw)")
    ax.set_ylabel("Calibrated probability")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        "raw_grid": raw_grid,
        "std_grid": std_grid,
        "ice_values": ice_values,
        "pdp": pdp,
        "sample_idx": sample_idx,
        "used_standardized": use_standardized_for_model,
    }


def plot_feature_importance(importances: np.ndarray, feature_names: list[str]) -> None:
    """
    Plot permutation feature importance as a sorted horizontal bar chart.

    Parameters
    ----------
    importances : array-like (d,)
        Importance scores for each feature.

    feature_names : list of str
        Names of the features, same length as importances.
    """

    importances = np.asarray(importances)
    feature_names = np.asarray(feature_names)

    # Sort from most important to least
    idx = np.argsort(importances)[::-1]
    sorted_imp = importances[idx]
    sorted_names = feature_names[idx]

    plt.figure(figsize=(10, 6))
    _ = plt.barh(sorted_names, sorted_imp, color="steelblue")
    plt.xlabel("Increase in Log-Loss When Feature is Permuted")
    plt.title("Permutation Feature Importance (Log-Loss)")
    plt.gca().invert_yaxis()  # Most important at the top
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
