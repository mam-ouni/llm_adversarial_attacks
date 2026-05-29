"""
05_visualize_all_results.py
----------------------------
Reads all CSVs from results/tables/ and generates publication-quality figures.
Run AFTER 01, 02 (and optionally 03, 04).

Figures produced:
    accuracy_drop_by_attack.png       — grouped bar chart, clean vs attacked accuracy
    asr_by_model_and_attack.png       — ASR comparison across models
    semantic_similarity_threshold.png — USE similarity with 0.9 threshold line
    mcnemar_significance_heatmap.png  — p-value heatmap (model × attack)
    vulnerability_gap_curve.png       — attack success rate vs cosine constraint
    combined_dashboard.png            — all key metrics in one figure

Usage:
    python src/05_visualize_all_results.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLES_DIR, COLORS, MODEL_LABELS, MODEL_IDS, ATTACK_NAMES,
    COSINE_CONSTRAINTS
)
from utils import apply_thesis_style, save_figure

warnings.filterwarnings("ignore")

# ─── Data loading helpers ──────────────────────────────────────────────────────

def load_table(filename: str) -> pd.DataFrame | None:
    path = TABLES_DIR / filename
    if not path.exists():
        print(f"  [!] Missing: {filename} — skipping related figure")
        return None
    return pd.read_csv(path)


# ─── Figure 1: Accuracy Drop ──────────────────────────────────────────────────

def plot_accuracy_drop(df_summary: pd.DataFrame) -> None:
    apply_thesis_style()

    models  = df_summary["model"].unique()
    attacks = df_summary["attack"].unique()
    n_models  = len(models)
    n_attacks = len(attacks)

    x     = np.arange(n_models)
    width = 0.18
    offsets = np.linspace(-(n_attacks - 1) * width / 2, (n_attacks - 1) * width / 2, n_attacks)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Draw clean accuracy as a dotted baseline (same for all attacks of same model)
    for i, m in enumerate(models):
        row = df_summary[df_summary["model"] == m].iloc[0]
        ax.plot(
            [i - 0.3, i + 0.3],
            [row["clean_acc_mean"] * 100] * 2,
            color=COLORS["clean"], linestyle="--", linewidth=1.4, alpha=0.7,
            label="Clean accuracy" if i == 0 else "_nolegend_"
        )

    # Draw attacked accuracy bars per attack
    for j, attack in enumerate(attacks):
        df_a    = df_summary[df_summary["attack"] == attack]
        vals    = []
        errs    = []
        for m in models:
            row = df_a[df_a["model"] == m]
            if len(row) == 0:
                vals.append(0); errs.append(0)
            else:
                vals.append(row["attacked_acc_mean"].values[0] * 100)
                errs.append(row["attacked_acc_std"].values[0]  * 100)

        bars = ax.bar(
            x + offsets[j], vals, width,
            label=attack.upper(),
            color=COLORS.get(attack, "#888780"),
            yerr=errs, capsize=3,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            edgecolor="white", linewidth=0.5, alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Model Accuracy — Clean vs Under Adversarial Attack\n"
        "(mean ± std, n=3 seeds, max 2 words modified)",
        fontsize=12, pad=10
    )
    ax.legend(fontsize=10, frameon=False, loc="upper right")

    save_figure(fig, "accuracy_drop_by_attack")


# ─── Figure 2: Attack Success Rate ────────────────────────────────────────────

def plot_asr(df_summary: pd.DataFrame) -> None:
    apply_thesis_style()

    models  = df_summary["model"].unique()
    attacks = df_summary["attack"].unique()
    n_models  = len(models)
    n_attacks = len(attacks)

    x       = np.arange(n_models)
    width   = 0.25
    offsets = np.linspace(-(n_attacks - 1) * width / 2, (n_attacks - 1) * width / 2, n_attacks)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for j, attack in enumerate(attacks):
        df_a = df_summary[df_summary["attack"] == attack]
        vals = []
        errs = []
        for m in models:
            row = df_a[df_a["model"] == m]
            vals.append(row["asr_mean"].values[0] * 100 if len(row) else 0)
            errs.append(row["asr_std"].values[0]  * 100 if len(row) else 0)

        ax.bar(
            x + offsets[j], vals, width,
            label=attack.upper(),
            color=COLORS.get(attack, "#888780"),
            yerr=errs, capsize=3,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Attack Success Rate by Model and Attack Type", fontsize=12, pad=10)
    ax.legend(fontsize=10, frameon=False)

    save_figure(fig, "asr_by_model_and_attack")


# ─── Figure 3: Semantic Similarity ────────────────────────────────────────────

def plot_semantic_similarity(df_summary: pd.DataFrame) -> None:
    apply_thesis_style()

    combos  = df_summary["model"] + " + " + df_summary["attack"].str.upper()
    vals    = df_summary["use_sim_mean"].values
    errs    = df_summary["use_sim_std"].values
    bar_colors = [
        COLORS.get(row["model"], "#888780")
        for _, row in df_summary.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.barh(
        combos, vals, xerr=errs,
        color=bar_colors, alpha=0.85,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        edgecolor="white", linewidth=0.5, height=0.55
    )

    # Stealth threshold line at 0.9
    ax.axvline(
        0.9, color=COLORS["threshold"], linestyle="--",
        linewidth=1.5, label="Stealth threshold (0.90)"
    )

    # Value labels
    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            val - 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="right", fontsize=9,
            color="white" if val > 0.5 else "#2C2C2A"
        )

    ax.set_xlabel("Mean USE Cosine Similarity", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_title(
        "Semantic Similarity of Adversarial Examples\n"
        "Below 0.90 = detectable by semantic similarity filter",
        fontsize=12, pad=10
    )
    ax.legend(fontsize=10, frameon=False)
    ax.invert_yaxis()

    save_figure(fig, "semantic_similarity_threshold")


# ─── Figure 4: McNemar heatmap ────────────────────────────────────────────────

def plot_mcnemar_heatmap(df_mcnemar: pd.DataFrame) -> None:
    apply_thesis_style()

    models  = sorted(df_mcnemar["model"].unique())
    attacks = sorted(df_mcnemar["attack"].unique())

    pval_matrix = np.ones((len(models), len(attacks)))
    for _, row in df_mcnemar.iterrows():
        i = models.index(row["model"])
        j = attacks.index(row["attack"])
        pval_matrix[i, j] = row["p_value"]

    # Show −log10(p) for visibility
    log_p = -np.log10(np.clip(pval_matrix, 1e-10, 1.0))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(log_p, cmap="Blues", aspect="auto", vmin=0)

    ax.set_xticks(range(len(attacks)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([a.upper() for a in attacks], fontsize=11)
    ax.set_yticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in models], fontsize=10)

    for i in range(len(models)):
        for j in range(len(attacks)):
            p     = pval_matrix[i, j]
            sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            color = "white" if log_p[i, j] > 3 else "#2C2C2A"
            ax.text(j, i, f"p={p:.3f}\n{sig}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("−log₁₀(p)", fontsize=9)

    ax.set_title("McNemar Test Significance\n(attack vs no attack)", fontsize=12, pad=10)
    fig.tight_layout()

    save_figure(fig, "mcnemar_significance_heatmap")


# ─── Figure 5: Vulnerability gap curve ────────────────────────────────────────

def plot_vulnerability_gap(df_raw: pd.DataFrame) -> None:
    """
    Simulates accuracy vs cosine constraint by filtering df_raw to only
    include adversarial examples above a given USE similarity threshold.
    Shows that tighter constraints (higher similarity) → lower ASR.
    """
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model_key in df_raw["model"].unique():
        df_m    = df_raw[df_raw["model"] == model_key]
        asrs    = []
        for constraint in COSINE_CONSTRAINTS:
            df_filt = df_m[df_m["use_similarity"] >= constraint]
            if len(df_filt) == 0:
                asrs.append(0)
            else:
                # Under this constraint, only count successes where similarity ≥ constraint
                successes = df_filt["attack_success"].sum()
                total     = len(df_m)  # denominator is ALL examples, not filtered
                asrs.append(successes / total * 100)

        ax.plot(
            COSINE_CONSTRAINTS, asrs,
            marker="o", markersize=5, linewidth=1.8,
            color=COLORS.get(model_key, "#888780"),
            label=MODEL_LABELS.get(model_key, model_key).replace("\n", " ")
        )

    ax.axvline(0.9, color=COLORS["threshold"], linestyle="--",
               linewidth=1.2, alpha=0.7, label="Stealth threshold (0.90)")

    ax.set_xlabel("Cosine Similarity Constraint", fontsize=11)
    ax.set_ylabel("Effective Attack Success Rate (%)", fontsize=11)
    ax.set_title(
        "Vulnerability Gap — ASR vs Semantic Similarity Constraint\n"
        "Higher constraint = harder to fool the model undetectably",
        fontsize=12, pad=10
    )
    ax.legend(fontsize=10, frameon=False)
    ax.set_ylim(0, 100)
    ax.invert_xaxis()

    save_figure(fig, "vulnerability_gap_curve")


# ─── Figure 6: Combined dashboard ─────────────────────────────────────────────

def plot_dashboard(df_summary: pd.DataFrame) -> None:
    """
    2×2 dashboard combining: accuracy drop, ASR, USE similarity, McNemar p-values.
    One-page overview for thesis appendix.
    """
    apply_thesis_style()

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    models  = list(df_summary["model"].unique())
    attacks = list(df_summary["attack"].unique())
    x       = np.arange(len(models))
    width   = 0.25
    offsets = [-width / 2, width / 2]

    # ── Top-left: accuracy drop ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for j, attack in enumerate(attacks):
        df_a = df_summary[df_summary["attack"] == attack]
        vals = [df_a[df_a["model"] == m]["attacked_acc_mean"].values[0] * 100
                if len(df_a[df_a["model"] == m]) else 0 for m in models]
        ax1.bar(x + offsets[j], vals, width,
                label=attack.upper(), color=COLORS.get(attack, "#888780"),
                edgecolor="white", linewidth=0.4, alpha=0.9)
    for i, m in enumerate(models):
        clean = df_summary[df_summary["model"] == m]["clean_acc_mean"].values[0] * 100
        ax1.plot([i - 0.3, i + 0.3], [clean] * 2, "--",
                 color=COLORS["clean"], linewidth=1.5,
                 label="Clean acc" if i == 0 else "_nolegend_")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=8)
    ax1.set_ylabel("Accuracy (%)", fontsize=9)
    ax1.set_title("Accuracy under attack", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_ylim(0, 105)

    # ── Top-right: USE similarity ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    combos = [f"{m}\n{a.upper()}" for _, (m, a) in
              df_summary[["model", "attack"]].iterrows()]
    vals   = df_summary["use_sim_mean"].values
    colors = [COLORS.get(r["model"], "#888780") for _, r in df_summary.iterrows()]
    ax2.barh(combos, vals, color=colors, alpha=0.85, edgecolor="white",
             linewidth=0.4, height=0.55)
    ax2.axvline(0.9, color=COLORS["threshold"], linestyle="--",
                linewidth=1.2, label="0.90 threshold")
    ax2.set_xlabel("USE similarity", fontsize=9)
    ax2.set_title("Semantic similarity", fontsize=10)
    ax2.set_xlim(0, 1.05)
    ax2.legend(fontsize=8, frameon=False)
    ax2.invert_yaxis()

    # ── Bottom-left: ASR ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for j, attack in enumerate(attacks):
        df_a = df_summary[df_summary["attack"] == attack]
        vals = [df_a[df_a["model"] == m]["asr_mean"].values[0] * 100
                if len(df_a[df_a["model"] == m]) else 0 for m in models]
        ax3.bar(x + offsets[j], vals, width,
                label=attack.upper(), color=COLORS.get(attack, "#888780"),
                edgecolor="white", linewidth=0.4, alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=8)
    ax3.set_ylabel("ASR (%)", fontsize=9)
    ax3.set_title("Attack success rate", fontsize=10)
    ax3.legend(fontsize=8, frameon=False)
    ax3.set_ylim(0, 105)

    # ── Bottom-right: % words changed ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for j, attack in enumerate(attacks):
        df_a = df_summary[df_summary["attack"] == attack]
        vals = [df_a[df_a["model"] == m]["pct_changed_mean"].values[0] * 100
                if len(df_a[df_a["model"] == m]) else 0 for m in models]
        ax4.bar(x + offsets[j], vals, width,
                label=attack.upper(), color=COLORS.get(attack, "#888780"),
                edgecolor="white", linewidth=0.4, alpha=0.9)
    ax4.set_xticks(x)
    ax4.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=8)
    ax4.set_ylabel("% words changed", fontsize=9)
    ax4.set_title("Perturbation rate", fontsize=10)
    ax4.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "XAI Adversarial Audit — Summary Dashboard",
        fontsize=14, fontweight="medium", y=1.01
    )

    save_figure(fig, "combined_dashboard")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from utils import setup_logger
    logger = setup_logger("05_visualize_all_results")
    logger.info("=" * 60)
    logger.info("Generating all result figures")
    logger.info("=" * 60)

    df_summary = load_table("statistical_summary.csv")
    df_mcnemar = load_table("mcnemar_tests.csv")
    df_raw     = load_table("attack_raw_results.csv")

    if df_summary is None:
        logger.error("statistical_summary.csv not found. Run 02_statistical_tests.py first.")
        return

    logger.info("Generating accuracy drop figure…")
    plot_accuracy_drop(df_summary)

    logger.info("Generating ASR figure…")
    plot_asr(df_summary)

    logger.info("Generating semantic similarity figure…")
    plot_semantic_similarity(df_summary)

    if df_mcnemar is not None:
        logger.info("Generating McNemar heatmap…")
        plot_mcnemar_heatmap(df_mcnemar)

    if df_raw is not None:
        logger.info("Generating vulnerability gap curve…")
        plot_vulnerability_gap(df_raw)

    logger.info("Generating combined dashboard…")
    plot_dashboard(df_summary)

    logger.info(f"\nAll figures saved to results/figures/")


if __name__ == "__main__":
    main()
