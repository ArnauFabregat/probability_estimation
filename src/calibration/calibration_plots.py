import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


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
