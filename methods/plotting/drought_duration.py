"""Box plot of drought spell durations across models."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import save_figure


def fig_drought_duration(
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
    hist_durations, _ = _extract_droughts(hist_series, threshold)

    model_durations = {"Historical": np.array(hist_durations)}
    for model_key, ensemble in ensembles.items():
        cfg = models_config[model_key]
        if cfg["frequency"] not in ("monthly", "daily"):
            continue
        all_durs = []
        for i in range(ensemble.metadata.n_realizations):
            df = ensemble.data_by_realization[i]
            col = min(site_index, df.shape[1] - 1)
            if cfg["frequency"] == "daily":
                series = df.iloc[:, col].resample("MS").sum()
            else:
                series = df.iloc[:, col]
            durs, _ = _extract_droughts(series.values, threshold)
            all_durs.extend(durs)
        model_durations[model_key] = np.array(all_durs)

    labels = list(model_durations.keys())
    data = [model_durations[k] for k in labels]
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
    ax.set_ylabel("Spell Duration (months)")
    ax.set_title("Drought Spell Durations (Q < 25th percentile)")
    save_figure(fig, filename, figure_dir)
