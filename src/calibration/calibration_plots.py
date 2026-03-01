import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curves_ax(ax, y_true, raw_probs, cal_probs, n_bins=10):
    """
    Plot calibration curves (raw vs calibrated) on the provided Axes.
    """
    frac_raw, mean_raw = calibration_curve(y_true, raw_probs, n_bins=n_bins)
    frac_cal, mean_cal = calibration_curve(y_true, cal_probs, n_bins=n_bins)

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


def plot_histograms_ax(ax, raw_probs, cal_probs, bins=40):
    """
    Plot probability histograms (raw vs calibrated) on the provided Axes.
    """
    ax.hist(raw_probs, bins=bins, alpha=0.5, label="Raw", density=True)
    ax.hist(cal_probs, bins=bins, alpha=0.5, label="Calibrated", density=True)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Probability Distribution")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)


def plot_raw_vs_calibrated_ax(ax, raw_probs, cal_probs):
    """
    Scatter of raw vs calibrated probabilities on the provided Axes.
    """
    ax.scatter(raw_probs, cal_probs, alpha=0.3, s=12)

    # Perfect identity line
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    ax.set_xlabel("Raw probability")
    ax.set_ylabel("Calibrated probability")
    ax.set_title("Raw vs Calibrated Mapping")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def calibration_diagnostics(y, raw_probs, cal_probs, n_bins=10, bins=40, suptitle="Calibration Diagnostics"):
    """
    Create a single figure with three subplots arranged in 1 row x 3 columns:
      [ Calibration Curve | Histograms | Raw vs Calibrated ]
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Unpack axes
    ax_cal, ax_hist, ax_scatter = axes

    # Draw each panel
    plot_calibration_curves_ax(ax_cal, y, raw_probs, cal_probs, n_bins=n_bins)
    plot_histograms_ax(ax_hist, raw_probs, cal_probs, bins=bins)
    plot_raw_vs_calibrated_ax(ax_scatter, raw_probs, cal_probs)

    # Optional overall title
    fig.suptitle(suptitle, fontsize=14, y=1.02)

    return fig, axes
