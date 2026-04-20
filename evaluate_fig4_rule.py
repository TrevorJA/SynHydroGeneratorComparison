"""
Pre-registered evaluation of the Figure 4 demotion rule.

This script makes the Fig 4 placement decision mechanical and
auditable. Running it post-HPC produces a single artifact
(outputs/cross_region/fig4_demotion_decision.json) plus a printed
verdict stating whether Figure 4 (MARE vs rank-score consistency)
belongs in the main text or the supplementary material.

## The rule (from notes/decisions_log.md 2026-04-17 fig4-demotion-rule)

Demote Figure 4 to the supplement IFF BOTH of:
  (a) Spearman rho(mean_mare, mean_rank_score) >= 0.9 across regimes,
      computed per regime, and
  (b) the top-3 generators by mean MARE match the top-3 by mean rank
      score in every regime.

If either condition fails in any regime, Figure 4 stays in the main
text because MARE and rank-score disagree somewhere and that
disagreement is informative.

Inputs:
  outputs/cross_region/mare.csv          -- region, model, mare
  outputs/cross_region/rank_scores.csv   -- region, model, mare_rank

Output:
  outputs/cross_region/fig4_demotion_decision.json
  stdout verdict with per-regime breakdown

Usage:
  python evaluate_fig4_rule.py
  python evaluate_fig4_rule.py --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)

RHO_THRESHOLD = 0.9
TOP_K = 3
DECISION_PATH = OUTPUT_DIR / "cross_region" / "fig4_demotion_decision.json"


def _per_regime_spearman(mare_df: pd.DataFrame, rank_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime Spearman rho between mean MARE and mean rank."""
    # Average each model's MARE and rank across any duplicates within a region
    mare_mean = (
        mare_df.groupby(["region", "model"])["mare"]
        .mean()
        .reset_index(name="mean_mare")
    )
    rank_mean = (
        rank_df.groupby(["region", "model"])["mare_rank"]
        .mean()
        .reset_index(name="mean_rank")
    )
    merged = pd.merge(mare_mean, rank_mean, on=["region", "model"], how="inner")
    merged = merged.dropna(subset=["mean_mare", "mean_rank"])

    rows: list[dict] = []
    for region, grp in merged.groupby("region"):
        if len(grp) < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(grp["mean_mare"], grp["mean_rank"])
        rows.append(
            {
                "region": region,
                "n_models": len(grp),
                "spearman_rho": float(rho) if rho == rho else np.nan,
                "p_value": float(pval) if pval == pval else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _per_regime_top_k_agreement(
    mare_df: pd.DataFrame, rank_df: pd.DataFrame, k: int = TOP_K
) -> pd.DataFrame:
    """For each regime, check whether top-k by MARE matches top-k by rank."""
    mare_mean = (
        mare_df.groupby(["region", "model"])["mare"]
        .mean()
        .reset_index(name="mean_mare")
    )
    rank_mean = (
        rank_df.groupby(["region", "model"])["mare_rank"]
        .mean()
        .reset_index(name="mean_rank")
    )
    merged = pd.merge(mare_mean, rank_mean, on=["region", "model"], how="inner")
    merged = merged.dropna(subset=["mean_mare", "mean_rank"])

    rows: list[dict] = []
    for region, grp in merged.groupby("region"):
        top_mare = set(grp.nsmallest(k, "mean_mare")["model"])
        top_rank = set(grp.nsmallest(k, "mean_rank")["model"])
        rows.append(
            {
                "region": region,
                "top_mare": sorted(top_mare),
                "top_rank": sorted(top_rank),
                "sets_match": top_mare == top_rank,
                "overlap": sorted(top_mare & top_rank),
                "mare_only": sorted(top_mare - top_rank),
                "rank_only": sorted(top_rank - top_mare),
            }
        )
    return pd.DataFrame(rows)


def evaluate() -> dict:
    """Evaluate the rule. Returns a decision dict."""
    mare_path = OUTPUT_DIR / "cross_region" / "mare.csv"
    rank_path = OUTPUT_DIR / "cross_region" / "rank_scores.csv"

    for p in (mare_path, rank_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run assemble_results.py first")

    mare_df = pd.read_csv(mare_path)
    rank_df = pd.read_csv(rank_path)

    rho_df = _per_regime_spearman(mare_df, rank_df)
    topk_df = _per_regime_top_k_agreement(mare_df, rank_df, k=TOP_K)

    all_rho_pass = bool(
        rho_df["spearman_rho"].dropna().ge(RHO_THRESHOLD).all()
        and not rho_df["spearman_rho"].isna().all()
    )
    all_topk_pass = bool(topk_df["sets_match"].all())
    demote = all_rho_pass and all_topk_pass

    decision = {
        "rule_version": "2026-04-17",
        "rho_threshold": RHO_THRESHOLD,
        "top_k": TOP_K,
        "all_regimes_rho_above_threshold": all_rho_pass,
        "all_regimes_top_k_match": all_topk_pass,
        "decision": "SUPPLEMENT" if demote else "MAIN",
        "per_regime_rho": rho_df.to_dict(orient="records"),
        "per_regime_top_k": topk_df.to_dict(orient="records"),
    }
    return decision


def _print_verdict(decision: dict, verbose: bool = False) -> None:
    print("=" * 70)
    print("Figure 4 demotion rule (pre-registered 2026-04-17)")
    print("=" * 70)
    print(f"Rule: demote IFF (rho >= {decision['rho_threshold']} in every regime)")
    print(f"      AND (top-{decision['top_k']} match in every regime)")
    print()
    print(f"  Spearman rho pass: {decision['all_regimes_rho_above_threshold']}")
    print(f"  Top-k match pass:  {decision['all_regimes_top_k_match']}")
    print()
    print(f"VERDICT: Figure 4 -> {decision['decision']}")
    print("=" * 70)

    if verbose:
        print("\nPer-regime Spearman rho:")
        for row in decision["per_regime_rho"]:
            rho = row["spearman_rho"]
            rho_str = f"{rho:.3f}" if rho == rho else "NaN"
            print(
                f"  {row['region']:25s}  " f"rho={rho_str:>6s}  (n={row['n_models']})"
            )
        print("\nPer-regime top-k agreement:")
        for row in decision["per_regime_top_k"]:
            status = "MATCH" if row["sets_match"] else "DIFFER"
            print(f"  {row['region']:25s}  {status}")
            print(f"    by MARE: {row['top_mare']}")
            print(f"    by rank: {row['top_rank']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-regime details",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        decision = evaluate()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISION_PATH, "w") as f:
        json.dump(decision, f, indent=2, default=str)
    logger.info("Saved decision to %s", DECISION_PATH)

    _print_verdict(decision, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
