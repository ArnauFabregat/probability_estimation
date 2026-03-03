from typing import Any, Dict, Optional, Sequence, Literal
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import brier_score_loss


def get_best_f1(y_true: ArrayLike, y_proba: ArrayLike, num_thresholds: int = 200
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


def ice_pdp_plot(
    model: object,
    X_std: np.ndarray,                 # standardized features → for NN models
    X_raw: np.ndarray,                 # original features → for axis/grid (and trees)
    feature_name: str,
    all_vars: Sequence[str],
    num_points: int = 50,
    n_samples: Optional[int] = None,
    mode: Literal["ice", "pdp", "both"] = "both",
    calibrator: Optional[object] = None,
    figsize: tuple[int, int] = (8, 5),
    random_state: int = 42,
    grid_percentiles: tuple[float, float] = (0.5, 99.5),
    model_input_space: Literal["auto", "raw", "standardized"] = "auto",
) -> Dict[str, Any]:
    """
    ICE/PDP with careful handling of model input space and calibrators.

    Key behaviors:
    - Grid/axis are *always in original (raw) units* for interpretability.
    - The model can be fed raw or standardized inputs, controlled by `model_input_space`:
        * "standardized" → feed X_std (recommended for (Monotonic)NNs)
        * "raw"          → feed X_raw (tree/boosted trees)
        * "auto"         → legacy heuristic (temperature calibrator ⇒ standardized)
    - Calibration:
        * Temperature scaling → requires logits (model.predict_logits).
        * Isotonic/Platt     → expects probabilities; robustly supports calibrators
          exposing `.predict`, `.transform`, or `.predict_proba`.

    Returns a dict with raw/std grids, ICE matrix, PDP, and sample indices.
    """

    # --- Helper: detect temperature scaling ---
    def is_temperature_scaler(cal) -> bool:
        if cal is None:
            return False
        m = getattr(cal, "method", None)
        if isinstance(m, str) and m.lower() == "temperature":
            return True
        return hasattr(cal, "temperature")

    # --- Decide model input representation ---
    if model_input_space == "standardized":
        use_standardized_for_model = True
    elif model_input_space == "raw":
        use_standardized_for_model = False
    else:
        # "auto" fallback (legacy behavior):
        # - If temperature calibrator is used, we assume a NN path (standardized).
        # - Otherwise raw. (Override by passing model_input_space explicitly.)
        use_standardized_for_model = is_temperature_scaler(calibrator)

    # --- 1) Find feature index ---
    if feature_name not in all_vars:
        raise ValueError(f"{feature_name} not in all_vars.")
    idx = all_vars.index(feature_name)

    # --- 2) Build grid in ORIGINAL units and also STANDARDIZED version ---
    raw_col = X_raw[:, idx]
    low, high = np.percentile(raw_col, list(grid_percentiles))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Grid percentiles produced non-finite bounds. Check data.")
    if low == high:
        # Expand slightly if degenerate to avoid zero-width grid
        eps = 1e-6 if low == 0 else abs(low) * 1e-6
        low, high = low - eps, high + eps

    raw_grid = np.linspace(low, high, num_points)  # original units

    # Standardize with dataset mean/std (fallback if std==0)
    mean = raw_col.mean()
    std = raw_col.std(ddof=0) if raw_col.shape[0] > 1 else 1.0
    std = (std if std != 0 else 1.0)
    std_grid = (raw_grid - mean) / std

    # --- 3) Sample rows ---
    N = X_std.shape[0]
    if n_samples is None or n_samples >= N:
        sample_idx = np.arange(N)
    else:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(N, size=n_samples, replace=False)

    # --- Helpers for prediction/calibration ---
    def _to_1d(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a.ravel()
        return a.reshape(-1)

    def get_raw_probs(X_eval: np.ndarray) -> np.ndarray:
        """
        Returns P(y=1). Tries predict_proba; falls back to predict if needed.
        """
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(X_eval))
            # common shapes: (n,2) or (n,)
            if p.ndim == 2:
                if p.shape[1] == 2:
                    return p[:, 1]
                # If multiclass, default to first column unless specified (could be extended)
                return p[:, 0]
            return _to_1d(p)

        # Fallback: some NNs expose .predict returning probabilities
        if hasattr(model, "predict"):
            p = np.asarray(model.predict(X_eval))
            return _to_1d(p)

        raise AttributeError(
            "Model must implement predict_proba(X) or predict(X) returning probabilities in [0,1]."
        )

    def _apply_probability_calibrator(prob_1d: np.ndarray) -> np.ndarray:
        """
        Apply Platt/isotonic-like calibrator on probabilities (shape (n,)).
        Supports .predict, .transform, or .predict_proba.
        """
        prob_1d = _to_1d(prob_1d)
        cal = calibrator

        # Preferred: .predict (e.g., sklearn IsotonicRegression)
        if hasattr(cal, "predict"):
            return _to_1d(cal.predict(prob_1d))

        # Sometimes calibrators expose a transform API
        if hasattr(cal, "transform"):
            return _to_1d(cal.transform(prob_1d))

        # Some wrappers might offer predict_proba on 1-d scores
        if hasattr(cal, "predict_proba"):
            out = cal.predict_proba(prob_1d)
            out = np.asarray(out)
            if out.ndim == 2 and out.shape[1] == 2:
                return out[:, 1]
            return _to_1d(out)

        raise AttributeError(
            "Calibrator must implement predict(probs), transform(probs), or predict_proba(probs)."
        )

    def get_calibrated(X_eval: np.ndarray) -> np.ndarray:
        """
        Unified interface:
        - Temperature scaling → logits path.
        - Isotonic/Platt → probability path.
        """
        if calibrator is None:
            return get_raw_probs(X_eval)

        # Temperature scaling expects logits
        method = getattr(calibrator, "method", None)
        if method == "temperature" or hasattr(calibrator, "temperature"):
            if not hasattr(model, "predict_logits"):
                raise AttributeError(
                    "Temperature scaling requires model.predict_logits(X) to provide logits."
                )
            logits = np.asarray(model.predict_logits(X_eval)).reshape(-1)
            return _to_1d(calibrator.predict_proba(logits))

        # Otherwise (isotonic/Platt): pass probabilities through calibrator
        p_raw = get_raw_probs(X_eval)
        p_cal = _apply_probability_calibrator(p_raw)
        return _to_1d(p_cal)

    # --- 4) Compute ICE ---
    ice_values = np.zeros((len(sample_idx), num_points), dtype=float)

    for i, row_idx in enumerate(sample_idx):
        if use_standardized_for_model:
            base = X_std[row_idx].copy()
            X_eval = np.repeat(base.reshape(1, -1), num_points, axis=0)
            X_eval[:, idx] = std_grid
        else:
            base = X_raw[row_idx].copy()
            X_eval = np.repeat(base.reshape(1, -1), num_points, axis=0)
            X_eval[:, idx] = raw_grid

        ice_values[i] = get_calibrated(X_eval)

    # --- 5) PDP ---
    pdp = ice_values.mean(axis=0)

    # --- 6) Plot calibrated vs non‑calibrated side-by-side ---
    _, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    ##########################################
    # LEFT PLOT → NON‑CALIBRATED
    ##########################################
    ax = axes[0]
    # recompute ICE/PDP without calibrator
    ice_raw = np.zeros_like(ice_values)
    for i, row_idx in enumerate(sample_idx):
        if use_standardized_for_model:
            base = X_std[row_idx].copy()
            X_eval = np.repeat(base.reshape(1, -1), num_points, axis=0)
            X_eval[:, idx] = std_grid
        else:
            base = X_raw[row_idx].copy()
            X_eval = np.repeat(base.reshape(1, -1), num_points, axis=0)
            X_eval[:, idx] = raw_grid

        ice_raw[i] = get_raw_probs(X_eval)

    pdp_raw = ice_raw.mean(axis=0)

    # ICE lines
    if mode in ("ice", "both"):
        for i in range(len(sample_idx)):
            ax.plot(raw_grid, ice_raw[i], alpha=0.25, color="gray")

    # PDP
    if mode in ("pdp", "both"):
        ax.plot(raw_grid, pdp_raw, color="blue", linewidth=3, label="PDP (raw)")
        ax.legend()

    ax.set_title(f"Uncalibrated — {feature_name}")
    ax.set_xlabel(f"{feature_name} (raw)")
    ax.set_ylabel("Predicted probability")
    ax.grid(True)

    ##########################################
    # RIGHT PLOT → CALIBRATED
    ##########################################
    ax = axes[1]

    # ICE (already computed)
    if mode in ("ice", "both"):
        for i in range(len(sample_idx)):
            ax.plot(raw_grid, ice_values[i], alpha=0.25, color="gray")

    # PDP
    if mode in ("pdp", "both"):
        ax.plot(raw_grid, pdp, color="red", linewidth=3, label="PDP (calibrated)")
        ax.legend()

    ax.set_title(f"Calibrated — {feature_name}")
    ax.set_xlabel(f"{feature_name} (raw)")
    ax.set_ylabel("Calibrated probability")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        "raw_grid": raw_grid,        # original units (x-axis)
        "std_grid": std_grid,        # standardized values fed to NN if selected
        "ice_values": ice_values,    # [n_samples_used, num_points]
        "pdp": pdp,                  # [num_points]
        "sample_idx": sample_idx,    # indices of rows used
        "used_standardized": use_standardized_for_model,
    }
