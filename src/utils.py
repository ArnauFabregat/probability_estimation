import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


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
