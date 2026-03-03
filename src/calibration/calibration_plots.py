from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sklearn.calibration import calibration_curve


def plot_calibration_curves_ax(
    ax: Axes,
    y_true: ArrayLike,
    raw_probs: ArrayLike,
    cal_probs: ArrayLike,
    n_bins: int = 10,
) -> None:
    """
    Plot raw vs calibrated calibration curves on the provided Matplotlib Axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes to draw on.
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0/1).
    raw_probs : array-like of shape (n_samples,)
        Uncalibrated predicted probabilities.
    cal_probs : array-like of shape (n_samples,)
        Calibrated predicted probabilities.
    n_bins : int, default=10
        Number of bins to use for computing calibration curves.

    Returns
    -------
    None
        Draws on `ax` in-place.
    """
    y_arr = np.asarray(y_true, dtype=float).reshape(-1)
    raw_arr = np.asarray(raw_probs, dtype=float).reshape(-1)
    cal_arr = np.asarray(cal_probs, dtype=float).reshape(-1)

    frac_raw, mean_raw = calibration_curve(y_arr, raw_arr, n_bins=n_bins)
    frac_cal, mean_cal = calibration_curve(y_arr, cal_arr, n_bins=n_bins)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    # Raw
    ax.plot(mean_raw, frac_raw, "s-", lw=1.5, label="Raw model")

    # Calibrated
    ax.plot(mean_cal, frac_cal, "o-", lw=1.5, label="Calibrated")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration / Reliability Curve")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)


def plot_histograms_ax(
    ax: Axes,
    raw_probs: ArrayLike,
    cal_probs: ArrayLike,
    bins: int = 40,
) -> None:
    """
    Plot probability distributions (histograms) for raw and calibrated probabilities.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes to draw on.
    raw_probs : array-like of shape (n_samples,)
        Uncalibrated predicted probabilities.
    cal_probs : array-like of shape (n_samples,)
        Calibrated predicted probabilities.
    bins : int, default=40
        Number of histogram bins.

    Returns
    -------
    None
    """
    raw_arr = np.asarray(raw_probs, dtype=float).reshape(-1)
    cal_arr = np.asarray(cal_probs, dtype=float).reshape(-1)

    ax.hist(raw_arr, bins=bins, alpha=0.5, label="Raw", density=True)
    ax.hist(cal_arr, bins=bins, alpha=0.5, label="Calibrated", density=True)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Probability Distribution")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)


def plot_raw_vs_calibrated_ax(
    ax: Axes,
    raw_probs: ArrayLike,
    cal_probs: ArrayLike,
) -> None:
    """
    Plot a scatter comparison of raw vs calibrated probabilities.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes to draw on.
    raw_probs : array-like of shape (n_samples,)
        Uncalibrated predicted probabilities.
    cal_probs : array-like of shape (n_samples,)
        Calibrated predicted probabilities.

    Returns
    -------
    None
    """
    raw_arr = np.asarray(raw_probs, dtype=float).reshape(-1)
    cal_arr = np.asarray(cal_probs, dtype=float).reshape(-1)

    ax.scatter(raw_arr, cal_arr, alpha=0.3, s=12)

    # Identity line
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    ax.set_xlabel("Raw probability")
    ax.set_ylabel("Calibrated probability")
    ax.set_title("Raw vs Calibrated Mapping")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def calibration_diagnostics(
    y: ArrayLike,
    raw_probs: ArrayLike,
    cal_probs: ArrayLike,
    n_bins: int = 10,
    bins: int = 40,
    suptitle: str = "Calibration Diagnostics",
) -> Tuple[Figure, tuple[Axes, Axes, Axes]]:
    """
    Create a calibration diagnostic figure with three subplots:

    1. Calibration curve (raw vs calibrated)
    2. Histogram comparison
    3. Raw vs calibrated scatter plot

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Ground-truth binary labels.
    raw_probs : array-like of shape (n_samples,)
        Uncalibrated predicted probabilities.
    cal_probs : array-like of shape (n_samples,)
        Calibrated predicted probabilities.
    n_bins : int, default=10
        Number of bins for calibration_curve().
    bins : int, default=40
        Number of histogram bins.
    suptitle : str, default="Calibration Diagnostics"
        Title of the entire figure.

    Returns
    -------
    fig : Figure
        The created Matplotlib Figure.
    axes : tuple[Axes, Axes, Axes]
        Tuple of three Axes: (calibration, histogram, scatter).
    """
    fig, axes_arr = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Guarantee a tuple of three Axes for a precise return type
    ax_cal: Axes = axes_arr[0]
    ax_hist: Axes = axes_arr[1]
    ax_scatter: Axes = axes_arr[2]
    axes_tuple = (ax_cal, ax_hist, ax_scatter)

    plot_calibration_curves_ax(ax_cal, y, raw_probs, cal_probs, n_bins=n_bins)
    plot_histograms_ax(ax_hist, raw_probs, cal_probs, bins=bins)
    plot_raw_vs_calibrated_ax(ax_scatter, raw_probs, cal_probs)

    fig.suptitle(suptitle, fontsize=14, y=1.02)

    return fig, axes_tuple
