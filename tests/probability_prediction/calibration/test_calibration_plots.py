from unittest.mock import Mock, patch

import numpy as np

from probability_prediction.calibration.calibration_plots import (
    calibration_diagnostics,
    plot_calibration_curves_ax,
    plot_histograms_ax,
    plot_raw_vs_calibrated_ax,
)


def test_plot_calibration_curves_ax_basic():
    np.random.seed(42)
    mock_ax = Mock()
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    raw_probs = np.random.rand(n_samples)
    cal_probs = np.clip(raw_probs + np.random.randn(n_samples) * 0.1, 0, 1)
    plot_calibration_curves_ax(ax=mock_ax, y_true=y_true, raw_probs=raw_probs, cal_probs=cal_probs, n_bins=10)
    assert mock_ax.plot.called
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called
    assert mock_ax.set_title.called
    assert mock_ax.grid.called
    assert mock_ax.legend.called
    mock_ax.set_xlabel.assert_called_with("Predicted probability")
    mock_ax.set_ylabel.assert_called_with("Observed frequency")
    mock_ax.set_title.assert_called_with("Calibration / Reliability Curve")


def test_plot_calibration_curves_ax_different_bins():
    np.random.seed(42)
    mock_ax = Mock()
    n_samples = 200
    y_true = np.random.randint(0, 2, n_samples)
    raw_probs = np.random.rand(n_samples)
    cal_probs = np.random.rand(n_samples)
    plot_calibration_curves_ax(ax=mock_ax, y_true=y_true, raw_probs=raw_probs, cal_probs=cal_probs, n_bins=5)
    assert mock_ax.plot.call_count == 3
    mock_ax.reset_mock()
    plot_calibration_curves_ax(ax=mock_ax, y_true=y_true, raw_probs=raw_probs, cal_probs=cal_probs, n_bins=20)
    assert mock_ax.plot.call_count == 3


def test_plot_calibration_curves_ax_list_inputs():
    mock_ax = Mock()
    y_true = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    raw_probs = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15, 0.75, 0.25]
    cal_probs = [0.85, 0.75, 0.25, 0.15, 0.65, 0.35, 0.80, 0.20, 0.70, 0.30]
    plot_calibration_curves_ax(ax=mock_ax, y_true=y_true, raw_probs=raw_probs, cal_probs=cal_probs, n_bins=5)
    assert mock_ax.plot.called
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called


def test_plot_histograms_ax_basic():
    np.random.seed(42)
    mock_ax = Mock()
    n_samples = 100
    raw_probs = np.random.rand(n_samples)
    cal_probs = np.clip(raw_probs + np.random.randn(n_samples) * 0.1, 0, 1)
    plot_histograms_ax(ax=mock_ax, raw_probs=raw_probs, cal_probs=cal_probs, bins=40)
    assert mock_ax.hist.call_count == 2
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called
    assert mock_ax.set_title.called
    assert mock_ax.legend.called
    assert mock_ax.grid.called
    mock_ax.set_xlabel.assert_called_with("Predicted probability")
    mock_ax.set_ylabel.assert_called_with("Density")
    mock_ax.set_title.assert_called_with("Probability Distribution")


def test_plot_histograms_ax_different_bins():
    np.random.seed(42)
    mock_ax = Mock()
    n_samples = 200
    raw_probs = np.random.rand(n_samples)
    cal_probs = np.random.rand(n_samples)
    plot_histograms_ax(ax=mock_ax, raw_probs=raw_probs, cal_probs=cal_probs, bins=20)
    assert mock_ax.hist.call_count == 2
    first_call = mock_ax.hist.call_args_list[0]
    assert "Raw" in str(first_call)
    assert first_call[1]["bins"] == 20
    assert first_call[1]["density"] is True
    mock_ax.reset_mock()
    plot_histograms_ax(ax=mock_ax, raw_probs=raw_probs, cal_probs=cal_probs, bins=100)
    assert mock_ax.hist.call_count == 2
    first_call = mock_ax.hist.call_args_list[0]
    assert first_call[1]["bins"] == 100


def test_plot_histograms_ax_list_inputs():
    mock_ax = Mock()
    raw_probs = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15, 0.75, 0.25]
    cal_probs = [0.85, 0.75, 0.25, 0.15, 0.65, 0.35, 0.80, 0.20, 0.70, 0.30]
    plot_histograms_ax(ax=mock_ax, raw_probs=raw_probs, cal_probs=cal_probs, bins=10)
    assert mock_ax.hist.call_count == 2
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called


def test_plot_histograms_ax_histogram_labels():
    np.random.seed(42)
    mock_ax = Mock()
    raw_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    cal_probs = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
    plot_histograms_ax(ax=mock_ax, raw_probs=raw_probs, cal_probs=cal_probs, bins=5)
    first_call = mock_ax.hist.call_args_list[0]
    assert "Raw" in str(first_call)
    assert first_call[1]["alpha"] == 0.5
    second_call = mock_ax.hist.call_args_list[1]
    assert "Calibrated" in str(second_call)
    assert second_call[1]["alpha"] == 0.5


def test_plot_raw_vs_calibrated_ax_basic():
    np.random.seed(42)
    mock_ax = Mock()
    n_samples = 100
    raw_probs = np.random.rand(n_samples)
    cal_probs = np.clip(raw_probs + np.random.randn(n_samples) * 0.1, 0, 1)
    plot_raw_vs_calibrated_ax(ax=mock_ax, raw_probs=raw_probs, cal_probs=cal_probs)
    assert mock_ax.scatter.called
    assert mock_ax.plot.called
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called
    assert mock_ax.set_title.called
    assert mock_ax.set_xlim.called
    assert mock_ax.set_ylim.called
    assert mock_ax.grid.called
    mock_ax.set_xlabel.assert_called_with("Raw probability")
    mock_ax.set_ylabel.assert_called_with("Calibrated probability")
    mock_ax.set_title.assert_called_with("Raw vs Calibrated Mapping")
    mock_ax.set_xlim.assert_called_with(0, 1)
    mock_ax.set_ylim.assert_called_with(0, 1)


def test_plot_raw_vs_calibrated_ax_list_inputs():
    mock_ax = Mock()
    raw_probs = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15, 0.75, 0.25]
    cal_probs = [0.85, 0.75, 0.25, 0.15, 0.65, 0.35, 0.80, 0.20, 0.70, 0.30]
    plot_raw_vs_calibrated_ax(ax=mock_ax, raw_probs=raw_probs, cal_probs=cal_probs)
    assert mock_ax.scatter.called
    assert mock_ax.plot.called
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called


def test_plot_raw_vs_calibrated_ax_scatter_params():
    np.random.seed(42)
    mock_ax = Mock()
    raw_probs = np.array([0.2, 0.4, 0.6, 0.8])
    cal_probs = np.array([0.25, 0.45, 0.65, 0.85])
    plot_raw_vs_calibrated_ax(ax=mock_ax, raw_probs=raw_probs, cal_probs=cal_probs)
    scatter_call = mock_ax.scatter.call_args
    assert scatter_call[1]["alpha"] == 0.3
    assert scatter_call[1]["s"] == 12


def test_calibration_diagnostics_basic():
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    raw_probs = np.random.rand(n_samples)
    cal_probs = np.clip(raw_probs + np.random.randn(n_samples) * 0.1, 0, 1)

    with (
        patch("probability_prediction.calibration.calibration_plots.plt.subplots") as mock_subplots,
        patch("probability_prediction.calibration.calibration_plots.plot_calibration_curves_ax") as mock_cal_curves,
        patch("probability_prediction.calibration.calibration_plots.plot_histograms_ax") as mock_histograms,
        patch("probability_prediction.calibration.calibration_plots.plot_raw_vs_calibrated_ax") as mock_scatter,
    ):
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        fig, axes = calibration_diagnostics(
            y=y_true, raw_probs=raw_probs, cal_probs=cal_probs, n_bins=10, bins=40, suptitle="Test Diagnostics"
        )

        mock_subplots.assert_called_once_with(1, 3, figsize=(18, 5), constrained_layout=True)
        mock_cal_curves.assert_called_once_with(mock_axes[0], y_true, raw_probs, cal_probs, n_bins=10)
        mock_histograms.assert_called_once_with(mock_axes[1], raw_probs, cal_probs, bins=40)
        mock_scatter.assert_called_once_with(mock_axes[2], raw_probs, cal_probs)
        mock_fig.suptitle.assert_called_once_with("Test Diagnostics", fontsize=14, y=1.02)
        assert fig is mock_fig
        assert axes == tuple(mock_axes)


def test_calibration_diagnostics_default_params():
    np.random.seed(42)
    y_true = np.array([1, 0, 1, 0, 1])
    raw_probs = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
    cal_probs = np.array([0.85, 0.25, 0.75, 0.35, 0.65])

    with (
        patch("probability_prediction.calibration.calibration_plots.plt.subplots") as mock_subplots,
        patch("probability_prediction.calibration.calibration_plots.plot_calibration_curves_ax") as mock_cal_curves,
        patch("probability_prediction.calibration.calibration_plots.plot_histograms_ax") as mock_histograms,
        patch("probability_prediction.calibration.calibration_plots.plot_raw_vs_calibrated_ax") as mock_scatter,
    ):
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        fig, axes = calibration_diagnostics(y=y_true, raw_probs=raw_probs, cal_probs=cal_probs)

        mock_subplots.assert_called_once_with(1, 3, figsize=(18, 5), constrained_layout=True)
        mock_cal_curves.assert_called_once_with(mock_axes[0], y_true, raw_probs, cal_probs, n_bins=10)
        mock_histograms.assert_called_once_with(mock_axes[1], raw_probs, cal_probs, bins=40)
        mock_scatter.assert_called_once_with(mock_axes[2], raw_probs, cal_probs)
        mock_fig.suptitle.assert_called_once_with("Calibration Diagnostics", fontsize=14, y=1.02)
        assert fig is mock_fig
        assert axes == tuple(mock_axes)


def test_calibration_diagnostics_custom_bins():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 50)
    raw_probs = np.random.rand(50)
    cal_probs = np.random.rand(50)

    with (
        patch("probability_prediction.calibration.calibration_plots.plt.subplots") as mock_subplots,
        patch("probability_prediction.calibration.calibration_plots.plot_calibration_curves_ax") as mock_cal_curves,
        patch("probability_prediction.calibration.calibration_plots.plot_histograms_ax") as mock_histograms,
        patch("probability_prediction.calibration.calibration_plots.plot_raw_vs_calibrated_ax") as mock_scatter,
    ):
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        fig, axes = calibration_diagnostics(y=y_true, raw_probs=raw_probs, cal_probs=cal_probs, n_bins=20, bins=50)

        mock_cal_curves.assert_called_once_with(mock_axes[0], y_true, raw_probs, cal_probs, n_bins=20)
        mock_histograms.assert_called_once_with(mock_axes[1], raw_probs, cal_probs, bins=50)
        mock_scatter.assert_called_once_with(mock_axes[2], raw_probs, cal_probs)


def test_calibration_diagnostics_list_inputs():
    y_true = [1, 1, 0, 0, 1, 0, 1, 0]
    raw_probs = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15]
    cal_probs = [0.85, 0.75, 0.25, 0.15, 0.65, 0.35, 0.80, 0.20]

    with (
        patch("probability_prediction.calibration.calibration_plots.plt.subplots") as mock_subplots,
        patch("probability_prediction.calibration.calibration_plots.plot_calibration_curves_ax") as mock_cal_curves,
        patch("probability_prediction.calibration.calibration_plots.plot_histograms_ax") as mock_histograms,
        patch("probability_prediction.calibration.calibration_plots.plot_raw_vs_calibrated_ax") as mock_scatter,
    ):
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        fig, axes = calibration_diagnostics(y=y_true, raw_probs=raw_probs, cal_probs=cal_probs)

        assert mock_cal_curves.called
        assert mock_histograms.called
        assert mock_scatter.called
        assert fig is mock_fig
        assert len(axes) == 3
