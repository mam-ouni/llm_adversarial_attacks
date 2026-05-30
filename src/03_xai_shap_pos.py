"""
03_xai_shap_pos.py
------------------
Detailed XAI analysis: SHAP token attributions × POS × semantic function.

Proves Hypothesis H1 — "Adjective Obsession":
    Models disproportionately rely on sentiment adjectives.

Also provides evidence for H2 — "Syntax Blindness":
    Negation words and modals receive near-zero SHAP importance.

Four levels of analysis:
    1. Broad POS (6 categories)  — overview figure
    2. Fine-grained POS (20+ tags) — detailed figure
    3. Functional categories       — negation / intensifier / modal vs adjective
    4. SHAP × VADER correlation    — model follows lexicon, not structure

Outputs:
    results/tables/shap_pos_scores.csv          — per-token SHAP + POS + VADER
    results/tables/shap_pos_broad.csv           — broad aggregation
    results/tables/shap_pos_finegrained.csv     — fine-grained aggregation
    results/tables/shap_functional_groups.csv   — negation / intensifier / modal
    results/tables/shap_top_tokens.csv          — top-50 most influential tokens
    results/figures/shap_pos_broad.png
    results/figures/shap_pos_finegrained.png
    results/figures/shap_functional_groups.png
    results/figures/shap_top_tokens.png
    results/figures/shap_vader_correlation.png
    results/figures/shap_token_examples.png

Usage:
    python src/03_xai_shap_pos.py
    python src/03_xai_shap_pos.py --n_samples 50   # quick run
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODEL_IDS, XAI_MODEL_KEY, SHAP_N_BACKGROUND, SHAP_N_SAMPLES,
    TABLES_DIR, COLORS, POS_ORDER, POS_CATEGORY_MAP,
    FINE_POS_MAP, FINE_POS_ORDER,
    NEGATION_WORDS, INTENSIFIER_WORDS,
    DATASET_NAME, DATASET_CONFIG,
)
from utils import (
    setup_logger, set_all_seeds, tokenize_with_pos,
    apply_thesis_style, save_figure
)

import torch
import shap
import nltk
from nltk import pos_tag, word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

warnings.filterwarnings("ignore")


# ─── Download VADER lexicon if needed ─────────────────────────────────────────
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
VADER = SentimentIntensityAnalyzer()


# ─── Model wrapper ────────────────────────────────────────────────────────────

def build_pipeline_fn(model, tokenizer, device):
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

def compute_shap_values(texts, predict_fn, logger):
    logger.info(f"  Building SHAP Partition explainer…")
    masker    = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(predict_fn, masker=masker, algorithm="partition")
    logger.info(f"  Computing SHAP for {len(texts)} texts…")
    return explainer(texts, fixed_context=1, batch_size=8)


# ─── Token classification helpers ─────────────────────────────────────────────

def get_functional_group(word: str, pos_tag_str: str) -> str:
    """
    Assign each token to a functional group for sentiment analysis:
        negation / intensifier / modal / adj_sentiment / other
    """
    w = word.lower().strip("'")
    if w in NEGATION_WORDS or w == "n't":
        return "negation"
    if w in INTENSIFIER_WORDS:
        return "intensifier"
    if pos_tag_str == "MD":
        return "modal"
    if pos_tag_str in ("JJ", "JJR", "JJS"):
        score = VADER.polarity_scores(word)["compound"]
        if score >= 0.1:
            return "adj_positive"
        if score <= -0.1:
            return "adj_negative"
        return "adj_neutral"
    if pos_tag_str in ("RB", "RBR", "RBS"):
        return "adverb"
    if pos_tag_str in ("NN", "NNS", "NNP", "NNPS"):
        return "noun"
    if pos_tag_str in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        return "verb"
    if pos_tag_str in ("DT", "IN", "CC", "TO", "PRP", "PRP$", "EX"):
        return "function_word"
    return "other"


def get_vader_score(word: str) -> float:
    return round(VADER.polarity_scores(word)["compound"], 4)


# ─── Main aggregation ─────────────────────────────────────────────────────────

def build_token_dataframe(texts, shap_values, logger):
    """
    Build a per-token DataFrame with:
        token, nltk_tag, broad_pos, fine_pos, functional_group,
        vader_score, shap_abs, shap_raw, text_idx
    """
    rows = []
    logger.info("  Tagging tokens and computing categories…")

    for i, text in enumerate(texts):
        shap_tokens = list(shap_values[i].data)
        shap_scores = shap_values[i].values

        if shap_scores.ndim == 2:
            raw_scores = shap_scores[:, 1]       # class 1 (positive)
        else:
            raw_scores = shap_scores

        abs_scores = np.abs(raw_scores)

        # NLTK POS tagging on original text
        nltk_tokens = word_tokenize(text)
        nltk_tags   = pos_tag(nltk_tokens)
        tag_dict    = {w.lower(): t for w, t in nltk_tags}

        for token, raw_val, abs_val in zip(shap_tokens, raw_scores, abs_scores):
            word      = str(token).strip()
            tag       = tag_dict.get(word.lower(), "NN")   # default NN if not found
            broad     = POS_CATEGORY_MAP.get(tag, "other")
            fine      = FINE_POS_MAP.get(tag, f"other ({tag})")
            func_grp  = get_functional_group(word, tag)
            vader     = get_vader_score(word)

            rows.append({
                "text_idx":        i,
                "token":           word,
                "nltk_tag":        tag,
                "broad_pos":       broad,
                "fine_pos":        fine,
                "functional_group": func_grp,
                "vader_score":     vader,
                "shap_abs":        round(float(abs_val), 6),
                "shap_raw":        round(float(raw_val), 6),
            })

    return pd.DataFrame(rows)


def aggregate_broad(df):
    agg = (
        df.groupby("broad_pos")["shap_abs"]
        .agg(mean_shap="mean", count="count", std="std")
        .reset_index()
    )
    total = agg["mean_shap"].sum()
    agg["importance_pct"] = (agg["mean_shap"] / total * 100).round(2)
    all_cats = pd.DataFrame({"broad_pos": POS_ORDER})
    agg = all_cats.merge(agg, on="broad_pos", how="left").fillna(0)
    return agg


def aggregate_finegrained(df):
    agg = (
        df.groupby("fine_pos")["shap_abs"]
        .agg(mean_shap="mean", count="count", std="std")
        .reset_index()
        .sort_values("mean_shap", ascending=False)
    )
    total = agg["mean_shap"].sum()
    agg["importance_pct"] = (agg["mean_shap"] / total * 100).round(2)
    return agg[agg["count"] >= 5]   # only categories with enough data


def aggregate_functional(df):
    func_order = [
        "adj_positive", "adj_negative", "adj_neutral",
        "intensifier", "adverb", "verb", "noun",
        "modal", "negation", "function_word", "other"
    ]
    agg = (
        df.groupby("functional_group")["shap_abs"]
        .agg(mean_shap="mean", count="count", std="std")
        .reset_index()
    )
    total = agg["mean_shap"].sum()
    agg["importance_pct"] = (agg["mean_shap"] / total * 100).round(2)
    all_cats = pd.DataFrame({"functional_group": func_order})
    agg = all_cats.merge(agg, on="functional_group", how="left").fillna(0)
    return agg


def get_top_tokens(df, n=30):
    top = (
        df.groupby("token")
        .agg(
            mean_shap=("shap_abs", "mean"),
            count=("shap_abs", "count"),
            broad_pos=("broad_pos", lambda x: x.mode()[0]),
            functional_group=("functional_group", lambda x: x.mode()[0]),
            vader_score=("vader_score", "mean"),
        )
        .reset_index()
        .query("count >= 3")                     # only tokens seen 3+ times
        .sort_values("mean_shap", ascending=False)
        .head(n)
    )
    return top


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_broad_pos(agg, model_key):
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    cats   = agg["broad_pos"].tolist()
    vals   = agg["importance_pct"].tolist()
    colors = [COLORS.get(c, "#B4B2A9") for c in cats]

    bars = ax.barh(cats, vals, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)

    ax.set_xlabel("Mean |SHAP| importance (%)", fontsize=11)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_title(
        f"XAI — Broad POS Importance | {model_key.upper()}\n"
        "H1: Adjectives dominate sentiment decisions",
        fontsize=12, pad=8
    )
    ax.invert_yaxis()
    save_figure(fig, "shap_pos_broad")


def plot_finegrained_pos(agg, model_key, top_n=20):
    apply_thesis_style()
    agg_top = agg.head(top_n)

    fig, ax = plt.subplots(figsize=(9, top_n * 0.42 + 1.5))

    color_map = {
        "adj":  "#7F77DD", "adv":  "#EF9F27", "verb": "#1D9E75",
        "noun": "#378ADD", "pron": "#888780",  "dete": "#B4B2A9",
        "prep": "#B4B2A9", "coor": "#B4B2A9",  "card": "#B4B2A9",
    }
    def tag_color(fine):
        for prefix, col in color_map.items():
            if fine.startswith(prefix):
                return col
        return "#B4B2A9"

    bars = ax.barh(
        agg_top["fine_pos"], agg_top["importance_pct"],
        color=[tag_color(t) for t in agg_top["fine_pos"]],
        height=0.65, edgecolor="white", linewidth=0.4
    )
    for bar, val in zip(bars, agg_top["importance_pct"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Mean |SHAP| importance (%)", fontsize=11)
    ax.set_xlim(0, agg_top["importance_pct"].max() * 1.2)
    ax.set_title(
        f"XAI — Fine-Grained POS Importance (top {top_n}) | {model_key.upper()}\n"
        "Breaking down adjective sub-types and verb forms",
        fontsize=12, pad=8
    )
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#7F77DD", label="Adjectives"),
        Patch(color="#EF9F27", label="Adverbs"),
        Patch(color="#1D9E75", label="Verbs"),
        Patch(color="#378ADD", label="Nouns"),
        Patch(color="#B4B2A9", label="Function words"),
    ]
    ax.legend(handles=legend, fontsize=9, frameon=False,
              loc="lower right")

    fig.tight_layout()
    save_figure(fig, "shap_pos_finegrained")


def plot_functional_groups(agg, model_key):
    """
    Key figure for H1 + H2:
    - Positive/negative adjectives → high importance (H1)
    - Negation words + modals → near-zero importance (H2)
    """
    apply_thesis_style()

    label_map = {
        "adj_positive":  "Adjectives — positive",
        "adj_negative":  "Adjectives — negative",
        "adj_neutral":   "Adjectives — neutral",
        "intensifier":   "Intensifiers (very, extremely…)",
        "adverb":        "Other adverbs",
        "verb":          "Verbs",
        "noun":          "Nouns",
        "modal":         "Modal verbs (would, could…)",
        "negation":      "Negation (not, never, n't…)",
        "function_word": "Function words",
        "other":         "Other",
    }
    color_map = {
        "adj_positive":  COLORS["sentiment_pos"],
        "adj_negative":  COLORS["sentiment_neg"],
        "adj_neutral":   "#888780",
        "intensifier":   COLORS["intensifier"],
        "adverb":        "#EF9F27",
        "verb":          COLORS["verb"],
        "noun":          COLORS["noun"],
        "modal":         COLORS["modal"],
        "negation":      COLORS["negation"],
        "function_word": COLORS["stopword"],
        "other":         COLORS["other"],
    }

    agg = agg[agg["importance_pct"] > 0].copy()
    agg["label"] = agg["functional_group"].map(label_map)

    fig, ax = plt.subplots(figsize=(9, len(agg) * 0.5 + 1.5))

    bars = ax.barh(
        agg["label"], agg["importance_pct"],
        color=[color_map.get(g, "#B4B2A9") for g in agg["functional_group"]],
        height=0.6, edgecolor="white", linewidth=0.4
    )
    for bar, val in zip(bars, agg["importance_pct"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)

    # Annotation for negation
    neg_row = agg[agg["functional_group"] == "negation"]
    if len(neg_row) > 0:
        neg_val = neg_row["importance_pct"].values[0]
        neg_idx = list(agg["functional_group"]).index("negation")
        ax.annotate(
            f"← H2: Negation ignored\n   ({neg_val:.1f}% vs adj {agg[agg['functional_group']=='adj_positive']['importance_pct'].values[0]:.1f}%)",
            xy=(neg_val, neg_idx),
            xytext=(neg_val + 2, neg_idx - 0.8),
            fontsize=8, color=COLORS["negation"],
            arrowprops=dict(arrowstyle="->", color=COLORS["negation"], lw=0.8)
        )

    ax.set_xlabel("Mean |SHAP| importance (%)", fontsize=11)
    ax.set_xlim(0, agg["importance_pct"].max() * 1.35)
    ax.set_title(
        f"XAI — Functional Group Importance | {model_key.upper()}\n"
        "H1: Sentiment adjectives dominate | H2: Negation near-zero",
        fontsize=12, pad=8
    )
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, "shap_functional_groups")


def plot_top_tokens(df_top, model_key):
    apply_thesis_style()

    color_fn = lambda grp: {
        "adj_positive": COLORS["sentiment_pos"],
        "adj_negative": COLORS["sentiment_neg"],
        "negation":     COLORS["negation"],
        "intensifier":  COLORS["intensifier"],
        "modal":        COLORS["modal"],
    }.get(grp, COLORS["stopword"])

    fig, ax = plt.subplots(figsize=(9, len(df_top) * 0.38 + 1.5))
    bars = ax.barh(
        df_top["token"], df_top["mean_shap"],
        color=[color_fn(g) for g in df_top["functional_group"]],
        height=0.65, edgecolor="white", linewidth=0.4
    )
    ax.set_xlabel("Mean |SHAP| score", fontsize=11)
    ax.set_title(
        f"Top-{len(df_top)} Most Influential Tokens | {model_key.upper()}\n"
        "Tokens driving sentiment decisions",
        fontsize=12, pad=8
    )
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend = [
        Patch(color=COLORS["sentiment_pos"], label="Positive adjective"),
        Patch(color=COLORS["sentiment_neg"], label="Negative adjective"),
        Patch(color=COLORS["negation"],      label="Negation word"),
        Patch(color=COLORS["intensifier"],   label="Intensifier"),
        Patch(color=COLORS["modal"],         label="Modal verb"),
        Patch(color=COLORS["stopword"],      label="Other"),
    ]
    ax.legend(handles=legend, fontsize=9, frameon=False)
    fig.tight_layout()
    save_figure(fig, "shap_top_tokens")


def plot_vader_correlation(df, model_key):
    """
    Scatter: VADER sentiment score vs SHAP importance.
    If the model learns the lexicon, adjectives with high |VADER| → high SHAP.
    Negation words: VADER ≈ 0, SHAP ≈ 0 → confirms syntax blindness.
    """
    apply_thesis_style()

    df_plot = df[df["broad_pos"].isin(["adjective", "adverb", "other"])].copy()
    df_plot = df_plot.sample(min(500, len(df_plot)), random_state=42)

    color_fn = lambda grp: {
        "adj_positive": COLORS["sentiment_pos"],
        "adj_negative": COLORS["sentiment_neg"],
        "negation":     COLORS["negation"],
        "intensifier":  COLORS["intensifier"],
        "modal":        COLORS["modal"],
        "adj_neutral":  "#888780",
    }.get(grp, "#B4B2A9")

    colors = [color_fn(g) for g in df_plot["functional_group"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_plot["vader_score"], df_plot["shap_abs"],
               c=colors, alpha=0.5, s=20, edgecolors="none")

    ax.axvline(0, color="#888780", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="#888780", linewidth=0.8, linestyle="--", alpha=0.5)

    # Regression line
    from numpy.polynomial.polynomial import polyfit
    valid = df_plot[["vader_score", "shap_abs"]].dropna()
    if len(valid) > 10:
        b, m = polyfit(valid["vader_score"], valid["shap_abs"], 1)
        x_line = np.linspace(valid["vader_score"].min(), valid["vader_score"].max(), 100)
        ax.plot(x_line, b + m * x_line, color="#2C2C2A", linewidth=1.5,
                linestyle="-", label=f"Trend (slope={m:.3f})")

    # Correlation coefficient
    corr = df_plot[["vader_score", "shap_abs"]].corr().iloc[0, 1]
    ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=11, color="#2C2C2A")

    ax.set_xlabel("VADER sentiment score (token-level)", fontsize=11)
    ax.set_ylabel("Mean |SHAP| importance", fontsize=11)
    ax.set_title(
        f"SHAP Importance × VADER Sentiment Score | {model_key.upper()}\n"
        "Do models follow the sentiment lexicon?",
        fontsize=12, pad=8
    )

    from matplotlib.patches import Patch
    legend = [
        Patch(color=COLORS["sentiment_pos"], label="Positive adj"),
        Patch(color=COLORS["sentiment_neg"], label="Negative adj"),
        Patch(color=COLORS["negation"],      label="Negation"),
        Patch(color=COLORS["intensifier"],   label="Intensifier"),
        Patch(color="#B4B2A9",               label="Other"),
    ]
    ax.legend(handles=legend, fontsize=9, frameon=False)
    fig.tight_layout()
    save_figure(fig, "shap_vader_correlation")


def plot_token_heatmaps(texts, shap_values, n=5):
    apply_thesis_style()
    n = min(n, len(texts))
    fig, axes = plt.subplots(n, 1, figsize=(12, n * 1.5 + 1))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax     = axes[i]
        tokens = list(shap_values[i].data)
        scores = shap_values[i].values
        vals   = scores[:, 1] if scores.ndim == 2 else scores

        valid  = [(t, v) for t, v in zip(tokens, vals)
                  if str(t).strip() not in ("[CLS]", "[SEP]", "<s>", "</s>")]
        if not valid:
            continue
        v_tok, v_val = zip(*valid)
        v_val = np.array(v_val)
        vmax  = max(abs(v_val).max(), 1e-6)
        norm  = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap  = plt.get_cmap("RdBu")

        x = 0
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        for tok, val in zip(v_tok, v_val):
            word  = str(tok).strip()
            w     = max(len(word) * 0.012 + 0.02, 0.04)
            color = cmap(norm(val))
            ax.add_patch(plt.Rectangle((x, 0.1), w, 0.8,
                         color=color, transform=ax.transAxes, clip_on=False))
            tc = "white" if abs(val) > vmax * 0.4 else "black"
            ax.text(x + w/2, 0.5, word, ha="center", va="center",
                    fontsize=9, color=tc, transform=ax.transAxes)
            x += w + 0.008
            if x > 0.97:
                break
        ax.set_title(f"Example {i+1}", fontsize=9, loc="left", pad=2)

    fig.suptitle("SHAP Token Heatmap (blue=positive, red=negative sentiment)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save_figure(fig, "shap_token_examples")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    model_key = args.model     or XAI_MODEL_KEY
    n_samples = args.n_samples or SHAP_N_SAMPLES

    logger = setup_logger("03_xai_shap_pos")
    logger.info("=" * 60)
    logger.info(f"XAI Detailed POS Analysis — {model_key} | n={n_samples}")
    logger.info("Levels: broad POS / fine-grained POS / functional / VADER")
    logger.info("=" * 60)

    set_all_seeds(42)

    # ── Load model ─────────────────────────────────────────────────────────────
    model_id  = MODEL_IDS[model_key]
    logger.info(f"Loading {model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # ── Load SST-2 ─────────────────────────────────────────────────────────────
    logger.info("Loading SST-2 validation…")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation")
    texts   = [ex["sentence"] for ex in dataset][:n_samples]
    logger.info(f"Using {len(texts)} examples")

    # ── SHAP ───────────────────────────────────────────────────────────────────
    predict_fn  = build_pipeline_fn(model, tokenizer, device)
    shap_values = compute_shap_values(texts, predict_fn, logger)

    # ── Build token DataFrame ──────────────────────────────────────────────────
    df_tokens = build_token_dataframe(texts[:len(shap_values)], shap_values, logger)
    df_tokens.to_csv(TABLES_DIR / "shap_pos_scores.csv", index=False)
    logger.info(f"  Token scores saved — {len(df_tokens)} rows")

    # ── Aggregations ───────────────────────────────────────────────────────────
    df_broad = aggregate_broad(df_tokens)
    df_broad.to_csv(TABLES_DIR / "shap_pos_broad.csv", index=False)

    df_fine = aggregate_finegrained(df_tokens)
    df_fine.to_csv(TABLES_DIR / "shap_pos_finegrained.csv", index=False)

    df_func = aggregate_functional(df_tokens)
    df_func.to_csv(TABLES_DIR / "shap_functional_groups.csv", index=False)

    df_top = get_top_tokens(df_tokens, n=30)
    df_top.to_csv(TABLES_DIR / "shap_top_tokens.csv", index=False)

    logger.info("All tables saved")

    # ── Log summary ────────────────────────────────────────────────────────────
    logger.info("\nBroad POS importance:")
    for _, r in df_broad.iterrows():
        bar = "█" * int(r["importance_pct"] / 2)
        logger.info(f"  {r['broad_pos']:<12} {r['importance_pct']:>6.1f}%  {bar}")

    logger.info("\nFunctional groups (top 5):")
    for _, r in df_func.sort_values("importance_pct", ascending=False).head(5).iterrows():
        logger.info(f"  {r['functional_group']:<20} {r['importance_pct']:>6.1f}%")

    neg_row = df_func[df_func["functional_group"] == "negation"]
    if len(neg_row) > 0:
        logger.info(f"\n  → Negation importance: {neg_row['importance_pct'].values[0]:.2f}% (H2 evidence)")

    # ── Figures ────────────────────────────────────────────────────────────────
    logger.info("\nGenerating figures…")
    plot_broad_pos(df_broad, model_key)
    plot_finegrained_pos(df_fine, model_key)
    plot_functional_groups(df_func, model_key)
    plot_top_tokens(df_top, model_key)
    plot_vader_correlation(df_tokens, model_key)
    plot_token_heatmaps(texts, shap_values, n=5)

    logger.info("Done — all figures saved to results/figures/")


if __name__ == "__main__":
    main()
