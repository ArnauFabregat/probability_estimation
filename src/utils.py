from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from typing import Optional, Sequence, Dict, Any
from sklearn.metrics import brier_score_loss


def get_best_f1(y_true: ArrayLike, y_proba: ArrayLike, num_thresholds: int = 200
                ) -> Tuple[float, float, np.ndarray, np.ndarray]:
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


def calculate_brier_metrics(y_true: ArrayLike, y_proba: ArrayLike) -> Tuple[float, float, float, float]:
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


def ice_pdp_plot_xgb_or_nn(
    model,
    X_std: np.ndarray,                 # standardized features → for NN models
    X_raw: np.ndarray,                 # original features → for axis/grid (and trees)
    feature_name: str,
    all_vars: Sequence[str],
    num_points: int = 50,
    n_samples: Optional[int] = None,
    mode: str = "both",                # "ice" | "pdp" | "both"
    calibrator: Optional[object] = None,
    plot_calibrated: bool = False,
    figsize=(8, 5),
    random_state: int = 42,
    grid_percentiles: tuple[float, float] = (0.5, 99.5),
) -> Dict[str, Any]:
    """
    Uses standardized inputs iff calibrator indicates temperature scaling;
    otherwise uses raw/original inputs. Plot x-axis is always in original units.
    """

    # --- 0) Decide which representation to feed the model ---
    def is_temperature_scaler(cal) -> bool:
        if cal is None:
            return False
        m = getattr(cal, "method", None)
        if isinstance(m, str) and m.lower() == "temperature":
            return True
        # Also support a logits temperature calibrator without .method
        return hasattr(cal, "temperature")

    use_standardized_for_model = is_temperature_scaler(calibrator)

    # --- 1) Find feature index ---
    if feature_name not in all_vars:
        raise ValueError(f"{feature_name} not in all_vars.")
    idx = all_vars.index(feature_name)

    # --- 2) Build grid in ORIGINAL units and (optionally) convert to STANDARDIZED ---
    raw_col = X_raw[:, idx]
    low, high = np.percentile(raw_col, list(grid_percentiles))
    raw_grid = np.linspace(low, high, num_points)  # original units

    # z = (x - mean) / std for the selected feature
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

    # --- Helpers ---
    def get_raw_probs(X_eval):
        p = model.predict_proba(X_eval)
        p = np.asarray(p)
        if p.ndim == 2:
            if p.shape[1] == 2:
                return p[:, 1]
            else:
                return p[:, 0]
        return p.reshape(-1)

    def get_calibrated(X_eval):
        if calibrator is None or not plot_calibrated:
            return get_raw_probs(X_eval)

        method = getattr(calibrator, "method", None)
        if method == "temperature" or hasattr(calibrator, "temperature"):
            # Temperature scaling expects LOGITS from the model
            logits = model.predict_logits(X_eval).reshape(-1)
            return calibrator.predict_proba(logits)

        # Platt / isotonic → pass probabilities
        p_raw = get_raw_probs(X_eval)
        p_cal = calibrator.predict_proba(p_raw)
        return np.asarray(p_cal).reshape(-1)

    # --- 4) Compute ICE ---
    ice_values = np.zeros((len(sample_idx), num_points), dtype=float)

    for i, row_idx in enumerate(sample_idx):
        if use_standardized_for_model:
            # NN / temperature calibrator path → standardized inputs
            base = X_std[row_idx].copy()
            X_eval = np.repeat(base.reshape(1, -1), num_points, axis=0)
            X_eval[:, idx] = std_grid
        else:
            # Trees / non-temperature path → raw/original inputs
            base = X_raw[row_idx].copy()
            X_eval = np.repeat(base.reshape(1, -1), num_points, axis=0)
            X_eval[:, idx] = raw_grid

        ice_values[i] = get_calibrated(X_eval)

    # --- 5) PDP ---
    pdp = ice_values.mean(axis=0)

    # --- 6) Plot (x-axis in ORIGINAL units) ---
    plt.figure(figsize=figsize)

    if mode in ("ice", "both"):
        for i in range(len(sample_idx)):
            plt.plot(raw_grid, ice_values[i], alpha=0.25, color="gray")

    if mode in ("pdp", "both"):
        label = "PDP (calibrated)" if (calibrator and plot_calibrated) else "PDP"
        plt.plot(raw_grid, pdp, color="red", linewidth=3, label=label)
        plt.legend()

    plt.xlabel(f"{feature_name} (original scale)")
    ylabel = "Calibrated probability" if (calibrator and plot_calibrated) else "Predicted probability"
    plt.ylabel(ylabel)
    plt.title(f"ICE/PDP for '{feature_name}' — samples: {len(sample_idx)}")
    plt.grid(True)
    plt.show()

    return {
        "raw_grid": raw_grid,      # original units (x-axis)
        "std_grid": std_grid,      # corresponding standardized values (for NN case)
        "ice_values": ice_values,  # [n_samples_used, num_points]
        "pdp": pdp,                # [num_points]
        "sample_idx": sample_idx
    }
