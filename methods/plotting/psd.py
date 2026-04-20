"""Power spectral density -- frequency domain comparison."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import HIST_STYLE, syn_style, save_figure


def fig_psd(
    model_data: dict,
    freq_label: str,
    period_unit: str,
    filename: str,
    model_colors: dict,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in model_data.items():
        vals = values - np.mean(values)
        n = len(vals)
        freqs = np.fft.rfftfreq(n, d=1.0)[1:]
        psd_vals = np.abs(np.fft.rfft(vals))[1:] ** 2 / n
        periods = 1.0 / freqs

        n_bins = min(80, len(periods) // 2)
        if n_bins < 5:
            continue
        bin_edges = np.logspace(
            np.log10(periods.min()), np.log10(periods.max()), n_bins + 1
        )
        smoothed_period, smoothed_psd = [], []
        for b in range(n_bins):
            mask = (periods >= bin_edges[b]) & (periods < bin_edges[b + 1])
            if mask.any():
                smoothed_period.append(np.exp(np.mean(np.log(periods[mask]))))
                smoothed_psd.append(np.exp(np.mean(np.log(psd_vals[mask]))))

        if name == "Historical":
            ax.plot(smoothed_period, smoothed_psd, **HIST_STYLE)
        else:
            ax.plot(smoothed_period, smoothed_psd, **syn_style(name, model_colors))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Period ({period_unit})")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(f"{freq_label} Power Spectrum")
    ax.legend()
    save_figure(fig, filename, figure_dir)
