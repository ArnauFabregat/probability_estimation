import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.calibration import calibration_curve


# TODO review - done with vibe coding
def get_best_f1(y_true, y_proba, num_thresholds=200):
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

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).reshape(-1)

    thresholds = np.linspace(0, 1, num_thresholds)
    f1_curve = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1_curve.append(f1_score(y_true, preds))

    f1_curve = np.array(f1_curve)
    best_idx = np.argmax(f1_curve)

    best_threshold = thresholds[best_idx]
    best_f1 = f1_curve[best_idx]

    return best_threshold, best_f1, f1_curve, thresholds


def ice_pdp_plot(
    model,
    X_raw,           # original-scale data (N, d)
    X_std,           # standardized data (N, d)
    feature_name,    # feature name
    all_vars,        # model variable order
    n_samples=None,  # None = use ALL samples, else subsample
    num_points=50,   # resolution
    mode="both",     # "ice", "pdp", or "both"
    figsize=(8, 5),
    random_state=42
):
    """
    Combined ICE + PDP (partial dependence) plot for a feature.

    Parameters
    ----------
    model : trained MonotonicNN
    X_raw : ndarray
        Raw/original-scale data
    X_std : ndarray
        Standardized data used for the model
    feature_name : str
        Feature for analysis
    all_vars : list[str]
        Feature names in model input order
    n_samples : int or None
        - None: use ALL samples for ICE curves
        - int : subsample that many individuals
    num_points : int
        Number of grid points
    mode : "ice", "pdp", or "both"
        Controls what is plotted
    figsize : tuple
        Size of the plot
    random_state : int
        RNG seed for reproducible subsampling

    Returns
    -------
    dict containing:
        "raw_grid": raw feature grid
        "std_grid": standardized grid
        "ice_values": array (n_samples, num_points)
        "pdp": array (num_points,)
        "sample_idx": sampled row indices
    """

    # --- 1. Feature index ---
    if feature_name not in all_vars:
        raise ValueError(f"{feature_name} not in all_vars.")
    idx = all_vars.index(feature_name)

    # --- 2. Raw grid ---
    raw_col = X_raw[:, idx]
    low, high = np.percentile(raw_col, [0.5, 99.5])
    raw_grid = np.linspace(low, high, num_points)

    # standardize grid (no scaler passed)
    mean = raw_col.mean()
    std = raw_col.std()
    std_grid = (raw_grid - mean) / std

    # --- 3. Decide which samples to use ---
    N = X_raw.shape[0]

    if n_samples is None or n_samples >= N:
        # use ALL samples
        sample_idx = np.arange(N)
    else:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(N, size=n_samples, replace=False)

    # --- 4. Compute ICE values ---
    ice_values = np.zeros((len(sample_idx), num_points))

    for i, row_idx in enumerate(sample_idx):
        base_std = X_std[row_idx].copy()

        X_eval = np.repeat(base_std.reshape(1, -1), num_points, axis=0)
        X_eval[:, idx] = std_grid

        probs = model.predict_proba(X_eval).reshape(-1)
        ice_values[i] = probs

    # --- 5. PDP (mean of ICE) ---
    pdp = ice_values.mean(axis=0)

    # --- 6. Plot ---
    plt.figure(figsize=figsize)

    if mode in ("ice", "both"):
        for i in range(len(sample_idx)):
            plt.plot(raw_grid, ice_values[i], alpha=0.25, color="gray")

    if mode in ("pdp", "both"):
        plt.plot(raw_grid, pdp, color="red", linewidth=3, label="PDP (average ICE)")
        plt.legend()

    plt.xlabel(f"{feature_name} (raw scale)")
    plt.ylabel("Predicted probability")
    plt.title(f"ICE/PDP for '{feature_name}' — samples: {len(sample_idx)}")
    plt.grid(True)
    plt.show()

    return {
        "raw_grid": raw_grid,
        "std_grid": std_grid,
        "ice_values": ice_values,
        "pdp": pdp,
        "sample_idx": sample_idx
    }


def ice_pdp_plot_xgb_or_nn(
    model,
    X,                
    feature_name,
    all_vars,
    n_samples=None,
    num_points=50,
    mode="both",
    figsize=(8, 5),
    random_state=42,
    calibrator=None,          
    plot_calibrated=True     
):
    """
    ICE/PDP plot for:
      - XGBoost / sklearn models (predict_proba)
      - Neural nets when used with Calibrator(method='temperature')

    Only two calibration behaviors:
      * If calibrator.method in {"platt","isotonic"}:
            p_raw = model.predict_proba(X_eval)[:,1]
            p = calibrator.predict_proba(p_raw)
      * If calibrator.method == "temperature":
            p = calibrator.predict_proba(X_eval)
    """

    # --- 1. Find feature index ---
    if feature_name not in all_vars:
        raise ValueError(f"{feature_name} not in all_vars.")
    idx = all_vars.index(feature_name)

    # --- 2. Build raw grid ---
    raw_col = X[:, idx]
    low, high = np.percentile(raw_col, [0.5, 99.5])
    raw_grid = np.linspace(low, high, num_points)

    # --- 3. Sample rows ---
    N = X.shape[0]
    if n_samples is None or n_samples >= N:
        sample_idx = np.arange(N)
    else:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(N, size=n_samples, replace=False)

    # --- Helper: raw probs from model ---
    def get_raw_probs(X_eval):
        p = model.predict_proba(X_eval)
        if p.ndim == 2:
            return p[:, 1]
        return p

    # --- Helper: apply calibration depending on method ---
    def get_calibrated(X_eval):
        if calibrator is None or not plot_calibrated:
            return get_raw_probs(X_eval)

        method = calibrator.method

        # temperature → classifier not needed, calibrator consumes X directly
        if method == "temperature":
            p = calibrator.predict_proba(X_eval)
            if getattr(p, "ndim", 1) == 2:
                p = p[:, 1]
            return np.asarray(p)

        # platt / isotonic → calibrator expects raw prob vector
        p_raw = get_raw_probs(X_eval)
        p_cal = calibrator.predict_proba(p_raw)
        return np.asarray(p_cal).reshape(-1)

    # --- 4. Compute ICE ---
    ice_values = np.zeros((len(sample_idx), num_points))

    for i, row_idx in enumerate(sample_idx):
        base = X[row_idx].copy()
        X_eval = np.repeat(base.reshape(1, -1), num_points, axis=0)
        X_eval[:, idx] = raw_grid
        ice_values[i] = get_calibrated(X_eval)

    # --- 5. PDP ---
    pdp = ice_values.mean(axis=0)

    # --- 6. Plot ---
    plt.figure(figsize=figsize)

    if mode in ("ice", "both"):
        for i in range(len(sample_idx)):
            plt.plot(raw_grid, ice_values[i], alpha=0.25, color="gray")

    if mode in ("pdp", "both"):
        label = "PDP (calibrated)" if calibrator and plot_calibrated else "PDP"
        plt.plot(raw_grid, pdp, color="red", linewidth=3, label=label)
        plt.legend()

    plt.xlabel(f"{feature_name} (raw scale)")
    ylabel = "Calibrated probability" if (calibrator and plot_calibrated) else "Predicted probability"
    plt.ylabel(ylabel)
    plt.title(f"ICE/PDP for '{feature_name}' — samples: {len(sample_idx)}")
    plt.grid(True)
    plt.show()

    return {
        "raw_grid": raw_grid,
        "ice_values": ice_values,
        "pdp": pdp,
        "sample_idx": sample_idx
    }


def plot_calibration_curves(y_true, raw_probs, cal_probs, n_bins=10):

    frac_raw, mean_raw = calibration_curve(y_true, raw_probs, n_bins=n_bins)
    frac_cal, mean_cal = calibration_curve(y_true, cal_probs, n_bins=n_bins)

    plt.figure(figsize=(7, 6))

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # Raw
    plt.plot(mean_raw, frac_raw, "s-", label="Raw model")

    # Calibrated
    plt.plot(mean_cal, frac_cal, "o-", label="Calibrated")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration / Reliability Curve")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_histograms(raw_probs, cal_probs, bins=40):

    plt.figure(figsize=(7,5))

    plt.hist(raw_probs, bins=bins, alpha=0.5, label="Raw", density=True)
    plt.hist(cal_probs, bins=bins, alpha=0.5, label="Calibrated", density=True)

    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Probability Distribution: Raw vs Calibrated")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_raw_vs_calibrated(raw_probs, cal_probs):

    plt.figure(figsize=(6, 6))
    plt.scatter(raw_probs, cal_probs, alpha=0.3, s=12)

    # Perfect identity line
    plt.plot([0,1],[0,1], "k--")

    plt.xlabel("Raw probability")
    plt.ylabel("Calibrated probability")
    plt.title("Raw vs Calibrated Probability Mapping")
    plt.grid(True)
    plt.show()

def calibration_diagnostics(y, raw_probs, cal_probs):
    plot_calibration_curves(y, raw_probs, cal_probs)
    plot_histograms(raw_probs, cal_probs)
    plot_raw_vs_calibrated(raw_probs, cal_probs)
