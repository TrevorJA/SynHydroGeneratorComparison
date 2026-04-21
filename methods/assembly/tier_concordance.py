"""Tier 1 (monthly) vs Tier 2 (annual) MARE concordance artifact.

Quantifies the relationship between a generator's Tier 1 (monthly)
performance and its Tier 2 (annual) performance across regions. Used
in the Discussion to support or refute the claim that "annual skill
does not necessarily translate to monthly resolution."

Reads `outputs/cross_region/mare.csv` (produced by cross_region.assemble)
and writes `outputs/cross_region/tier_concordance.csv`.

See notes/decisions_log.md 2026-04-17 tier-concordance-check-added.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def build_tier_concordance(mare_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(region, model) Tier 1 vs Tier 2 MARE comparison.

    Parameters
    ----------
    mare_df : pd.DataFrame
        Long-format DataFrame with at minimum the columns `region`,
        `model`, `mare` (Tier 1 / primary MARE), and `annual_tier_mare`
        (Tier 2 MARE when applicable).

    Returns
    -------
    pd.DataFrame
        One row per (region, model) with columns
        `region, model, tier1_mare, tier2_mare, tier_delta,
         rank_tier1, rank_tier2, rank_delta`. Entries where either
        tier is NaN are dropped.
    """
    if "mare" not in mare_df.columns or "annual_tier_mare" not in mare_df.columns:
        raise KeyError("mare_df must contain 'mare' and 'annual_tier_mare' columns")

    rows = []
    for region_id, grp in mare_df.groupby("region"):
        grp = grp.dropna(subset=["mare", "annual_tier_mare"])
        if grp.empty:
            continue

        rank_t1 = grp["mare"].rank(method="average")
        rank_t2 = grp["annual_tier_mare"].rank(method="average")

        for idx, r in grp.iterrows():
            rows.append(
                {
                    "region": region_id,
                    "model": r["model"],
                    "tier1_mare": r["mare"],
                    "tier2_mare": r["annual_tier_mare"],
                    "tier_delta": r["annual_tier_mare"] - r["mare"],
                    "rank_tier1": rank_t1.loc[idx],
                    "rank_tier2": rank_t2.loc[idx],
                    "rank_delta": rank_t2.loc[idx] - rank_t1.loc[idx],
                }
            )

    return pd.DataFrame(rows)


def assemble_tier_concordance(cross_region_dir: Path) -> Path:
    """Write tier_concordance.csv and log global Spearman rho.

    Parameters
    ----------
    cross_region_dir : Path
        The cross-region output directory (CROSS_REGION_DIR from config).
        Expects `mare.csv` to already exist there.

    Returns
    -------
    Path
        Path to the written CSV, or an empty placeholder path if no data.
    """
    mare_path = cross_region_dir / "mare.csv"
    if not mare_path.exists():
        logger.warning(
            "tier_concordance: %s missing; run assemble_results.py first",
            mare_path,
        )
        return mare_path

    mare_df = pd.read_csv(mare_path)
    concordance = build_tier_concordance(mare_df)

    out_path = cross_region_dir / "tier_concordance.csv"
    if concordance.empty:
        logger.warning(
            "tier_concordance: no (region, model) pairs with both tiers present"
        )
        pd.DataFrame(
            columns=[
                "region",
                "model",
                "tier1_mare",
                "tier2_mare",
                "tier_delta",
                "rank_tier1",
                "rank_tier2",
                "rank_delta",
            ]
        ).to_csv(out_path, index=False)
        return out_path

    concordance.to_csv(out_path, index=False)
    logger.info(
        "Saved %s (%d rows across %d regions)",
        out_path,
        len(concordance),
        concordance["region"].nunique(),
    )

    # Global Spearman rho on (tier1_mare, tier2_mare) -- does annual skill
    # track monthly skill?
    if len(concordance) >= 3:
        rho, pval = spearmanr(concordance["tier1_mare"], concordance["tier2_mare"])
        logger.info(
            "Tier1 vs Tier2 MARE Spearman rho = %.3f (p = %.4f, n = %d)",
            rho,
            pval,
            len(concordance),
        )

        # Per-region spearman
        for region_id, grp in concordance.groupby("region"):
            if len(grp) >= 3:
                r_region, p_region = spearmanr(grp["tier1_mare"], grp["tier2_mare"])
                logger.info(
                    "  %s: rho = %.3f (p = %.4f, n = %d)",
                    region_id,
                    r_region,
                    p_region,
                    len(grp),
                )

    # Flag models where Tier 1 and Tier 2 ranks diverge strongly. A
    # large positive rank_delta means the model performs better annually
    # than monthly (aggregation hides the monthly-scale problems).
    divergent = concordance[concordance["rank_delta"].abs() >= 4]
    if not divergent.empty:
        logger.info("Rank divergence >= 4 in %d (region, model) pairs", len(divergent))

    return out_path
