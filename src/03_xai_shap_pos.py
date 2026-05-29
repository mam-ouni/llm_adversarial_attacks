"""
03_xai_shap_pos.py
------------------
Computes SHAP token-level attributions on clean SST-2 examples,
aggregates importance scores by POS category (adjectives, verbs, nouns…),
and generates the key figure for Hypothesis H1:

    "Models disproportionately rely on adjectives for sentiment decisions."

Outputs:
    results/tables/shap_pos_scores.csv        — raw per-token SHAP scores with POS
    results/tables/shap_pos_aggregated.csv    — mean importance per POS category
    results/figures/shap_pos_importance.png   — bar chart (main thesis figure)
    results/figures/shap_token_examples.png   — token-level heatmap on 5 examples

Usage:
    python src/03_xai_shap_pos.py
    python src/03_xai_shap_pos.py --model bert-base    # override config default
    python src/03_xai_shap_pos.py --n_samples 50       # quick run
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODEL_IDS, XAI_MODEL_KEY, SHAP_N_BACKGROUND, SHAP_N_SAMPLES,
    TABLES_DIR, COLORS, POS_ORDER, DATASET_NAME, DATASET_CONFIG
)
from utils import setup_logger, set_all_seeds, tokenize_with_pos, apply_thesis_style, save_figure

import torch
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

warnings.filterwarnings("ignore")


# ─── Model wrapper for SHAP ───────────────────────────────────────────────────

def build_pipeline_fn(model, tokenizer, device):
    """
    Returns a function that maps a list of texts to class-1 (positive) probabilities.
    SHAP requires a simple text → probability function.
    """
    import torch.nn.functional as F

    def predict_proba(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        results = []
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=128, padding=True
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            results.append(probs)
        return np.array(results)

    return predict_proba


# ─── SHAP computation ─────────────────────────────────────────────────────────

def compute_shap_values(
    texts: list[str],
    predict_fn,
    n_background: int,
    logger,
) -> tuple:
    """
    Compute SHAP values using the Partition explainer (best for NLP).
    Returns (explainer, shap_values) where shap_values[i] is a shap.Explanation.
    """
    logger.info(f"  Building SHAP Partition explainer on {n_background} background texts…")
    masker   = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(predict_fn, masker=masker, algorithm="partition")

    logger.info(f"  Computing SHAP values for {len(texts)} texts…")
    shap_values = explainer(texts, fixed_context=1, batch_size=8)
    return explainer, shap_values


# ─── POS aggregation ──────────────────────────────────────────────────────────

def aggregate_shap_by_pos(
    texts: list[str],
    shap_values,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each token in each text:
        - assign POS category via NLTK
        - record |SHAP value| for class 1 (positive sentiment)

    Returns:
        df_tokens     — per-token DataFrame (text, token, pos_tag, pos_category, shap_abs)
        df_aggregated — mean |SHAP| per POS category (normalized to sum to 1)
    """
    token_rows = []

    for i, text in enumerate(texts):
        # SHAP tokens for this example
        shap_tokens = shap_values[i].data          # list of strings
        shap_scores = shap_values[i].values        # shape (n_tokens, n_classes)

        # Absolute SHAP for class 1 (positive)
        if shap_scores.ndim == 2:
            abs_scores = np.abs(shap_scores[:, 1])
        else:
            abs_scores = np.abs(shap_scores)

        # POS-tag the original text (finer granularity than SHAP tokens)
        pos_tagged = tokenize_with_pos(text)

        # Match SHAP tokens to NLTK tokens (best-effort by word)
        for j, (token, score) in enumerate(zip(shap_tokens, abs_scores)):
            token_clean = token.strip().lower()
            # find best POS match
            matched_cat = "other"
            for word, tag, cat in pos_tagged:
                if word.lower() == token_clean:
                    matched_cat = cat
                    break

            token_rows.append({
                "text_idx":     i,
                "token":        token,
                "pos_category": matched_cat,
                "shap_abs":     round(float(score), 6),
            })

    df_tokens = pd.DataFrame(token_rows)

    # Aggregate: mean absolute SHAP per POS category
    df_agg = (
        df_tokens.groupby("pos_category")["shap_abs"]
        .agg(mean_abs_shap="mean", count="count", std="std")
        .reset_index()
    )

    # Normalize mean importance to sum to 100% (interpretable for figures)
    total = df_agg["mean_abs_shap"].sum()
    df_agg["importance_pct"] = (df_agg["mean_abs_shap"] / total * 100).round(2)

    # Ensure all POS categories are present
    all_cats = pd.DataFrame({"pos_category": POS_ORDER})
    df_agg   = all_cats.merge(df_agg, on="pos_category", how="left").fillna(0)
    df_agg["pos_category"] = pd.Categorical(df_agg["pos_category"],
                                             categories=POS_ORDER, ordered=True)
    df_agg = df_agg.sort_values("pos_category")

    logger.info("POS importance breakdown:")
    for _, row in df_agg.iterrows():
        bar = "█" * int(row["importance_pct"] / 2)
        logger.info(f"  {row['pos_category']:<12} {row['importance_pct']:>6.1f}%  {bar}")

    return df_tokens, df_agg


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_pos_importance(df_agg: pd.DataFrame, model_key: str) -> None:
    """
    Bar chart: mean |SHAP| importance per POS category.
    This is the core figure for Hypothesis H1.
    """
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    categories  = df_agg["pos_category"].tolist()
    importances = df_agg["importance_pct"].tolist()
    bar_colors  = [COLORS.get(cat, "#888780") for cat in categories]

    bars = ax.barh(categories, importances, color=bar_colors,
                   height=0.55, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, importances):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left",
            fontsize=10, color="#2C2C2A"
        )

    # Annotate the adjective bar
    adj_val = df_agg.loc[df_agg["pos_category"] == "adjective", "importance_pct"].values
    if len(adj_val) > 0 and adj_val[0] > 0:
        ax.axvline(adj_val[0], color=COLORS["adjective"], linestyle="--",
                   linewidth=1.2, alpha=0.5, label=f"Adjective level ({adj_val[0]:.1f}%)")
        ax.legend(fontsize=9, frameon=False)

    ax.set_xlabel("Mean |SHAP| importance (%)", fontsize=11)
    ax.set_xlim(0, max(importances) * 1.18)
    ax.set_title(
        f"XAI Feature Importance by POS Category — {model_key.upper()}\n"
        f"H1: Adjective obsession in sentiment analysis",
        fontsize=12, pad=10
    )
    ax.invert_yaxis()

    save_figure(fig, "shap_pos_importance")


def plot_token_heatmaps(
    texts: list[str],
    shap_values,
    n_examples: int = 5,
) -> None:
    """
    Token-level SHAP heatmap for n_examples sentences.
    Positive SHAP → positive sentiment signal (blue).
    Negative SHAP → negative sentiment signal (red).
    """
    apply_thesis_style()
    n = min(n_examples, len(texts))
    fig, axes = plt.subplots(n, 1, figsize=(12, n * 1.4 + 1))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax          = axes[i]
        tokens      = list(shap_values[i].data)
        scores      = shap_values[i].values

        # SHAP for class 1 (positive)
        if scores.ndim == 2:
            vals = scores[:, 1]
        else:
            vals = scores

        vmax   = max(abs(vals.max()), abs(vals.min()), 1e-6)
        norm   = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap   = plt.get_cmap("RdBu")

        x_pos  = 0
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        for token, val in zip(tokens, vals):
            word    = str(token).strip()
            w_width = max(len(word) * 0.012 + 0.02, 0.04)
            color   = cmap(norm(val))
            # draw colored box
            rect    = plt.Rectangle(
                (x_pos, 0.1), w_width, 0.8,
                color=color, transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)
            # text on top
            text_color = "white" if abs(val) > vmax * 0.4 else "black"
            ax.text(
                x_pos + w_width / 2, 0.5, word,
                ha="center", va="center", fontsize=9,
                color=text_color, transform=ax.transAxes
            )
            x_pos += w_width + 0.008
            if x_pos > 0.97:
                break

        ax.set_title(f"Example {i+1}", fontsize=9, loc="left", pad=2)

    fig.suptitle(
        "SHAP Token Attribution Heatmap\n(blue = positive signal, red = negative signal)",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    save_figure(fig, "shap_token_examples")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    model_key  = args.model    or XAI_MODEL_KEY
    n_samples  = args.n_samples or SHAP_N_SAMPLES

    logger = setup_logger("03_xai_shap_pos")
    logger.info("=" * 60)
    logger.info(f"XAI SHAP + POS Analysis — model: {model_key}")
    logger.info(f"Samples: {n_samples} | Background: {SHAP_N_BACKGROUND}")
    logger.info("=" * 60)

    set_all_seeds(42)

    # ── Load model ─────────────────────────────────────────────────────────────
    model_id  = MODEL_IDS[model_key]
    logger.info(f"Loading {model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # ── Load SST-2 validation set ──────────────────────────────────────────────
    logger.info("Loading SST-2 validation set…")
    dataset   = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation")
    texts     = [ex["sentence"] for ex in dataset][:n_samples]
    logger.info(f"Using {len(texts)} examples")

    # ── Build predict function and compute SHAP ────────────────────────────────
    predict_fn = build_pipeline_fn(model, tokenizer, device)
    _, shap_values = compute_shap_values(
        texts[:SHAP_N_BACKGROUND + 10],  # background + a few extras
        predict_fn,
        n_background=SHAP_N_BACKGROUND,
        logger=logger,
    )

    # ── Aggregate by POS ───────────────────────────────────────────────────────
    logger.info("Aggregating SHAP values by POS category…")
    df_tokens, df_agg = aggregate_shap_by_pos(texts[:len(shap_values)], shap_values, logger)

    # ── Save tables ────────────────────────────────────────────────────────────
    df_tokens.to_csv(TABLES_DIR / "shap_pos_scores.csv",      index=False)
    df_agg.to_csv(   TABLES_DIR / "shap_pos_aggregated.csv",  index=False)
    logger.info("Tables saved → shap_pos_scores.csv, shap_pos_aggregated.csv")

    # ── Generate figures ───────────────────────────────────────────────────────
    logger.info("Generating figures…")
    plot_pos_importance(df_agg, model_key)
    plot_token_heatmaps(texts, shap_values, n_examples=5)

    logger.info("Done.")


if __name__ == "__main__":
    main()
