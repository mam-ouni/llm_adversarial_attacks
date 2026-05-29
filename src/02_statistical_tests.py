"""
02_statistical_tests.py
------------------------
Reads attack_raw_results.csv (output of 01_run_attacks.py) and produces:

    results/tables/statistical_summary.csv   — ASR, USE sim, % words ± std (per model × attack)
    results/tables/mcnemar_tests.csv         — McNemar chi2, p-value, significance per combo
    results/tables/per_seed_results.csv      — Results broken down by seed (for ± std reporting)

This script contains NO plotting — pure statistics.
Run 05_visualize_all_results.py to generate figures from these tables.

Usage:
    python src/02_statistical_tests.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from config import TABLES_DIR, MODEL_IDS, ATTACK_NAMES, SEEDS
from utils import setup_logger, run_mcnemar, bootstrap_ci


# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_per_seed_metrics(df_group: pd.DataFrame) -> dict:
    """
    From one (model, attack, seed) group, compute:
        clean_acc, attacked_acc, asr, avg_use_sim, avg_pct_changed
    """
    clean_acc    = df_group["original_correct"].mean()
    attacked_acc = 1 - df_group["attack_success"].mean()  # acc after attack
    asr          = df_group["attack_success"].mean()
    use_sim      = df_group["use_similarity"].mean()
    pct_changed  = df_group["pct_words_changed"].mean()
    n            = len(df_group)

    return {
        "n_examples":    n,
        "clean_acc":     round(clean_acc,    4),
        "attacked_acc":  round(attacked_acc, 4),
        "acc_drop":      round(clean_acc - attacked_acc, 4),
        "asr":           round(asr,          4),
        "use_sim":       round(use_sim,       4),
        "pct_changed":   round(pct_changed,   4),
    }


def compute_confidence_interval(values: list[float]) -> tuple[float, float]:
    """
    95% CI via t-distribution (exact for n=3 seeds) AND bootstrap.
    Returns (mean, std, ci_lower, ci_upper) from t-distribution.
    """
    arr  = np.array(values)
    mean = np.mean(arr)
    std  = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    # 95% CI via t-distribution (df = n-1 = 2 for 3 seeds)
    if len(arr) > 1:
        se       = std / np.sqrt(len(arr))
        t_crit   = stats.t.ppf(0.975, df=len(arr) - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
    else:
        ci_lower = ci_upper = mean
    return mean, std, ci_lower, ci_upper


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger = setup_logger("02_statistical_tests")
    logger.info("=" * 60)
    logger.info("XAI Adversarial Attack Audit — Statistical Analysis")
    logger.info("=" * 60)

    # ── Load raw results ───────────────────────────────────────────────────────
    raw_path = TABLES_DIR / "attack_raw_results.csv"
    if not raw_path.exists():
        logger.error(f"Raw results not found: {raw_path}")
        logger.error("Run 01_run_attacks.py first.")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} records from {raw_path.name}")
    logger.info(f"Models: {df['model'].unique().tolist()}")
    logger.info(f"Attacks: {df['attack'].unique().tolist()}")
    logger.info(f"Seeds: {df['seed'].unique().tolist()}")

    # ── Per-seed breakdown ─────────────────────────────────────────────────────
    per_seed_rows = []
    for (model, attack, seed), grp in df.groupby(["model", "attack", "seed"]):
        metrics = compute_per_seed_metrics(grp)
        per_seed_rows.append({"model": model, "attack": attack, "seed": seed, **metrics})

    df_per_seed = pd.DataFrame(per_seed_rows)
    df_per_seed.to_csv(TABLES_DIR / "per_seed_results.csv", index=False)
    logger.info(f"Per-seed results saved → per_seed_results.csv")

    # ── Aggregate across seeds (mean ± std, 95% CI) ───────────────────────────
    summary_rows = []
    metric_cols  = ["clean_acc", "attacked_acc", "acc_drop", "asr", "use_sim", "pct_changed"]

    for (model, attack), grp in df_per_seed.groupby(["model", "attack"]):
        row = {"model": model, "attack": attack}

        for col in metric_cols:
            vals                     = grp[col].values
            mean, std, ci_lo, ci_hi  = compute_confidence_interval(vals)

            row[f"{col}_mean"]  = round(mean,  4)
            row[f"{col}_std"]   = round(std,   4)
            row[f"{col}_ci_lo"] = round(ci_lo, 4)
            row[f"{col}_ci_hi"] = round(ci_hi, 4)

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(TABLES_DIR / "statistical_summary.csv", index=False)
    logger.info("Statistical summary saved → statistical_summary.csv")

    # ── Pretty print for thesis table ─────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("Thesis-ready summary (mean ± std, n=3 seeds):")
    logger.info("─" * 60)

    header = f"{'Model':<12} {'Attack':<12} {'Clean acc':>10} {'Adv acc':>10} {'Drop':>8} {'ASR':>8} {'USE sim':>9}"
    logger.info(header)
    logger.info("─" * len(header))

    for _, row in df_summary.iterrows():
        logger.info(
            f"{row['model']:<12} {row['attack']:<12} "
            f"{row['clean_acc_mean']:>9.1%} "
            f"{row['attacked_acc_mean']:>9.1%} "
            f"{row['acc_drop_mean']:>7.1%} "
            f"{row['asr_mean']:>7.1%} "
            f"{row['use_sim_mean']:>8.3f}"
        )

    # ── McNemar tests ──────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("McNemar tests (pooled across seeds):")
    logger.info("─" * 60)

    mcnemar_rows = []

    for (model, attack), grp in df.groupby(["model", "attack"]):
        clean_correct = grp["original_correct"].astype(bool).tolist()
        adv_correct   = (~grp["attack_success"].astype(bool)).tolist()

        mc = run_mcnemar(clean_correct, adv_correct)

        sig_label = "***" if mc["p_value"] < 0.001 else \
                    "**"  if mc["p_value"] < 0.01  else \
                    "*"   if mc["p_value"] < 0.05  else "ns"

        logger.info(
            f"  {model:<12} × {attack:<12} | "
            f"b={mc['n10_b']:>4} c={mc['n01_c']:>4} | "
            f"χ²={mc['chi2']:>8.2f} | "
            f"p={mc['p_value']:.6f} {sig_label}"
        )

        mcnemar_rows.append({
            "model":        model,
            "attack":       attack,
            "n_concordant": mc["n11"] + mc["n00"],
            "n_b":          mc["n10_b"],
            "n_c":          mc["n01_c"],
            "chi2":         mc["chi2"],
            "p_value":      mc["p_value"],
            "significant":  mc["significant"],
            "sig_label":    sig_label,
        })

    df_mcnemar = pd.DataFrame(mcnemar_rows)
    df_mcnemar.to_csv(TABLES_DIR / "mcnemar_tests.csv", index=False)
    logger.info("\nMcNemar results saved → mcnemar_tests.csv")

    # ── Effect size (Cohen's h for proportions) ───────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("Effect size — Cohen's h (clean_acc vs attacked_acc):")
    logger.info("─" * 60)
    logger.info("  h > 0.2 small | h > 0.5 medium | h > 0.8 large")

    for _, row in df_summary.iterrows():
        p1    = row["clean_acc_mean"]
        p2    = row["attacked_acc_mean"]
        phi1  = 2 * np.arcsin(np.sqrt(p1))
        phi2  = 2 * np.arcsin(np.sqrt(p2))
        h     = abs(phi1 - phi2)
        label = "large" if h > 0.8 else "medium" if h > 0.5 else "small"
        logger.info(f"  {row['model']:<12} × {row['attack']:<12} | h = {h:.3f} ({label})")

    logger.info("\nAll statistical outputs saved to results/tables/")


if __name__ == "__main__":
    main()
