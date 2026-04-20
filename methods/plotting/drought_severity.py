"""Box plot of drought spell severities across models."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import save_figure


def fig_drought_severity(
    Q_monthly_hist: pd.DataFrame,
    ensembles: dict,
    models_config: dict,
    model_colors: dict,
    figure_dir: Path,
    filename: str,
    site_index: int = 0,
) -> None:
    from synhydro.core.validation._helpers import _extract_droughts

    hist_series = Q_monthly_hist.iloc[:, site_index].values
    threshold = np.percentile(hist_series, 25)
    _, hist_severities = _extract_droughts(hist_series, threshold)

    model_severities = {"Historical": np.array(hist_severities)}
    for model_key, ensemble in ensembles.items():
        cfg = models_config[model_key]
        if cfg["frequency"] not in ("monthly", "daily"):
            continue
        all_sevs = []
        for i in range(ensemble.metadata.n_realizations):
            df = ensemble.data_by_realization[i]
            col = min(site_index, df.shape[1] - 1)
            if cfg["frequency"] == "daily":
                series = df.iloc[:, col].resample("MS").sum()
            else:
                series = df.iloc[:, col]
            _, sevs = _extract_droughts(series.values, threshold)
            all_sevs.extend(sevs)
        model_severities[model_key] = np.array(all_sevs)

    labels = list(model_severities.keys())
    data = [model_severities[k] for k in labels]
    non_empty = [(l, d) for l, d in zip(labels, data) if len(d) > 0]
    if len(non_empty) < 2:
        return
    labels, data = zip(*non_empty)

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for i, label in enumerate(labels):
        color = (
            "lightgray" if label == "Historical" else model_colors.get(label, "gray")
        )
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(0.7)
    ax.set_ylabel("Cumulative Deficit (cms)")
    ax.set_title("Drought Spell Severity (Q < 25th percentile)")
    save_figure(fig, filename, figure_dir)
