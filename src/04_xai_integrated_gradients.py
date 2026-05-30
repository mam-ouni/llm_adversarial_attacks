"""
04_xai_integrated_gradients.py
--------------------------------
Uses Captum's Layer Integrated Gradients on the embedding layer to compute
token-level attributions for negation examples.

Proves Hypothesis H2 — "Syntax Blindness":
    "I don't think that this film is good"
    → the model ignores the negation prefix and decides based on "good"

Outputs:
    results/figures/ig_negation_heatmap.png       — attribution heatmap for all negation examples
    results/figures/ig_negation_bar_<i>.png        — per-example bar chart
    results/tables/ig_negation_attributions.csv    — raw attribution scores

Usage:
    python src/04_xai_integrated_gradients.py
    python src/04_xai_integrated_gradients.py --model bert-base
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
    MODEL_IDS, XAI_MODEL_KEY, NEGATION_EXAMPLES, IG_N_STEPS,
    TABLES_DIR, COLORS
)
from utils import setup_logger, set_all_seeds, apply_thesis_style, save_figure

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients

warnings.filterwarnings("ignore")


# ─── Attribution computation ──────────────────────────────────────────────────

def compute_ig_attributions(
    text: str,
    model,
    tokenizer,
    device: str,
    n_steps: int,
    target_label: int = 1,
) -> dict:
    """
    Compute Layer Integrated Gradients on the word embedding layer.

    Returns a dict with:
        tokens        — list of decoded sub-tokens
        attributions  — summed attribution per token (after L2 norm)
        pred_label    — model prediction (0 or 1)
        pred_prob     — confidence for the predicted class
    """
    model.eval()

    # ── Tokenize ───────────────────────────────────────────────────────────────
    encoding   = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=128)
    input_ids  = encoding["input_ids"].to(device)
    attn_mask  = encoding["attention_mask"].to(device)
    token_ids  = input_ids[0].tolist()
    tokens     = tokenizer.convert_ids_to_tokens(token_ids)

    # ── Baseline: all [PAD] token embeddings (standard IG baseline) ───────────
    baseline_ids = torch.zeros_like(input_ids)

    # ── Forward function returning logit for target_label ─────────────────────
    def forward_fn(inp_ids):
        out = model(input_ids=inp_ids, attention_mask=attn_mask)
        return out.logits[:, target_label]

    # ── LayerIG on word embedding layer ───────────────────────────────────────
    lig = LayerIntegratedGradients(forward_fn, model.base_model.embeddings.word_embeddings)

    attributions, delta = lig.attribute(
        inputs      = input_ids,
        baselines   = baseline_ids,
        n_steps     = n_steps,
        return_convergence_delta = True,
    )

    # Sum over embedding dimension → scalar per token
    attr_sum = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # ── Get prediction ─────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_label = int(np.argmax(probs))
    pred_prob  = float(probs[pred_label])

    return {
        "tokens":       tokens,
        "attributions": attr_sum,
        "pred_label":   pred_label,
        "pred_prob":    pred_prob,
        "delta":        float(delta.mean().abs()),
    }


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_negation_heatmap(all_results: list[dict]) -> None:
    """
    Multi-row heatmap showing token attributions for all negation examples.
    The key visual: [PAD] and negation prefix tokens should have near-zero attribution.
    """
    apply_thesis_style()

    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(13, n * 1.8 + 1.2))
    if n == 1:
        axes = [axes]

    label_map = {0: "NEG", 1: "POS"}

    for i, (ax, res) in enumerate(zip(axes, all_results)):
        tokens  = res["tokens"]
        attrs   = res["attributions"]
        label   = label_map[res["pred_label"]]
        prob    = res["pred_prob"]

        # Filter out [CLS] / [SEP]
        valid   = [(t, a) for t, a in zip(tokens, attrs)
                   if t not in ("[CLS]", "[SEP]", "<s>", "</s>")]
        v_tokens, v_attrs = zip(*valid) if valid else ([], [])

        v_tokens = list(v_tokens)
        v_attrs  = np.array(v_attrs, dtype=float)

        # Normalize per-example for visual comparison
        vmax = max(abs(v_attrs).max(), 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = plt.get_cmap("RdBu_r")

        # Draw as horizontal heatmap
        im = ax.imshow(
            v_attrs.reshape(1, -1),
            cmap=cmap, norm=norm,
            aspect="auto"
        )

        ax.set_xticks(range(len(v_tokens)))
        ax.set_xticklabels(v_tokens, rotation=35, ha="right", fontsize=9)
        ax.set_yticks([])

        verdict = f"Pred: {label} ({prob:.0%})"
        ax.set_ylabel(f"Ex {i+1}\n{verdict}", fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=80)

        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    fig.suptitle(
        "Integrated Gradients — Negation Examples\n"
        "H2: 'Syntax Blindness' — model ignores negation prefixes",
        fontsize=12, y=1.01
    )
    fig.tight_layout()
    save_figure(fig, "ig_negation_heatmap")


def plot_single_example_bar(res: dict, example_idx: int) -> None:
    """
    Bar chart for one example: positive attribution = pushes toward 'positive',
    negative = pushes toward 'negative'. Shows the adjective dominating.
    """
    apply_thesis_style()

    tokens = [t for t in res["tokens"] if t not in ("[CLS]", "[SEP]", "<s>", "</s>")]
    attrs  = [a for t, a in zip(res["tokens"], res["attributions"])
              if t not in ("[CLS]", "[SEP]", "<s>", "</s>")]

    colors = [COLORS["clean"] if v > 0 else COLORS["attacked"] for v in attrs]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(len(tokens)), attrs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#444441", linewidth=0.8)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Attribution score", fontsize=11)

    label_map = {0: "Negative", 1: "Positive"}
    ax.set_title(
        f"Integrated Gradients — Example {example_idx + 1}\n"
        f"Predicted: {label_map[res['pred_label']]} ({res['pred_prob']:.0%})",
        fontsize=12, pad=8
    )

    # Add legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color=COLORS["clean"],   label="→ Positive sentiment"),
        Patch(color=COLORS["attacked"], label="→ Negative sentiment"),
    ]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")

    save_figure(fig, f"ig_negation_bar_{example_idx + 1}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def plot_minimal_pair(pair_results: list[dict], pairs: list[dict],
                      model_key: str) -> None:
    """
    Side-by-side bar chart for a minimal pair.
    Shows how attribution changes when a single 'not' is added.

    Key visual for H2: if the model understood syntax, the two charts
    would look mirror-opposite. If it's blind to syntax, they look similar.
    """
    apply_thesis_style()

    label_map = {0: "Negative", 1: "Positive"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)

    for ax, res, pair in zip(axes, pair_results, pairs):
        # Filter special tokens
        tokens = [t for t in res["tokens"]
                  if t not in ("[CLS]", "[SEP]", "<s>", "</s>")]
        attrs  = [a for t, a in zip(res["tokens"], res["attributions"])
                  if t not in ("[CLS]", "[SEP]", "<s>", "</s>")]

        bar_colors = [COLORS["clean"] if v > 0 else COLORS["attacked"]
                      for v in attrs]

        bars = ax.bar(range(len(tokens)), attrs,
                      color=bar_colors, edgecolor="white",
                      linewidth=0.5, width=0.65)

        ax.axhline(0, color="#444441", linewidth=0.8)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=40, ha="right", fontsize=10)
        ax.set_ylabel("Attribution score", fontsize=10)

        pred_str     = label_map[res["pred_label"]]
        expected_str = pair["expected_sentiment"].capitalize()
        match        = "✓" if pred_str.lower() == pair["expected_sentiment"] else "✗"

        ax.set_title(
            f"{pair['label']}\n"
            f"\"{pair['text']}\"\n"
            f"Expected: {expected_str} | Predicted: {pred_str} {match} ({res['pred_prob']:.0%})",
            fontsize=10, pad=8,
            color="#085041" if match == "✓" else "#791F1F"
        )

        # Highlight the token that differs between the two (the added "not")
        for j, tok in enumerate(tokens):
            if tok.lower() in ("not", "n't", "nt") and pair["label"] == "Double negation":
                ax.get_children()[j].set_edgecolor("#E24B4A")
                ax.get_children()[j].set_linewidth(2)

    from matplotlib.patches import Patch
    handles = [
        Patch(color=COLORS["clean"],    label="→ Positive signal"),
        Patch(color=COLORS["attacked"], label="→ Negative signal"),
    ]
    fig.legend(handles=handles, fontsize=9, frameon=False,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Minimal Pair Analysis — Integrated Gradients | {model_key.upper()}\n"
        "H2: Does adding 'not' change the model's decision?",
        fontsize=12, y=1.02, fontweight="medium"
    )
    fig.tight_layout()
    save_figure(fig, "ig_minimal_pair_comparison")


def run_minimal_pair_analysis(model, tokenizer, device, n_steps, logger):
    """
    Run IG on the minimal pair and log the comparison.
    """
    from config import MINIMAL_PAIRS
    label_map = {0: "negative", 1: "positive"}

    logger.info("\n" + "=" * 60)
    logger.info("MINIMAL PAIR ANALYSIS")
    logger.info("=" * 60)
    logger.info("Testing: does adding one 'not' change the model's decision?")

    pair_results = []
    pair_rows    = []

    for pair in MINIMAL_PAIRS:
        logger.info(f"\n  [{pair['label']}]")
        logger.info(f"  Text     : \"{pair['text']}\"")
        logger.info(f"  Expected : {pair['expected_sentiment']}")

        res = compute_ig_attributions(
            pair["text"], model, tokenizer, device, n_steps, target_label=1
        )
        pair_results.append(res)

        pred_str  = label_map[res["pred_label"]]
        match     = "✓ CORRECT" if pred_str == pair["expected_sentiment"] else "✗ WRONG"
        logger.info(f"  Predicted: {pred_str} ({res['pred_prob']:.2%})  →  {match}")

        ranked = sorted(
            zip(res["tokens"], res["attributions"]),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        logger.info("  Top tokens:")
        for tok, val in ranked:
            bar = "▓" * int(abs(val) / max(abs(res["attributions"])) * 20)
            logger.info(f"    {tok:<15} {val:+.4f}  {bar}")

        for tok, val in zip(res["tokens"], res["attributions"]):
            pair_rows.append({
                "pair_label":  pair["label"],
                "text":        pair["text"],
                "expected":    pair["expected_sentiment"],
                "predicted":   pred_str,
                "correct":     int(pred_str == pair["expected_sentiment"]),
                "token":       tok,
                "attribution": round(float(val), 6),
            })

    # Key comparison log
    logger.info("\n" + "─" * 60)
    logger.info("COMPARISON SUMMARY:")
    for pair, res in zip(MINIMAL_PAIRS, pair_results):
        pred  = label_map[res["pred_label"]]
        match = "✓" if pred == pair["expected_sentiment"] else "✗"
        logger.info(
            f"  {pair['label']:<25} → pred={pred:<10} expected={pair['expected_sentiment']:<10} {match}"
        )

    if pair_results[0]["pred_label"] == pair_results[1]["pred_label"]:
        logger.info(
            "\n  → BOTH PREDICTED THE SAME CLASS — strong H2 evidence:"
            "\n    adding 'not' did NOT change the model's decision."
        )
    else:
        logger.info(
            "\n  → Different predictions — model partially handles double negation."
        )

    # Save CSV
    pd.DataFrame(pair_rows).to_csv(
        TABLES_DIR / "ig_minimal_pair.csv", index=False
    )
    logger.info("  Saved → ig_minimal_pair.csv")

    return pair_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--pair_only", action="store_true",
                        help="Run only the minimal pair analysis (faster)")
    args = parser.parse_args()

    model_key = args.model   or XAI_MODEL_KEY
    n_steps   = args.n_steps or IG_N_STEPS

    logger = setup_logger("04_xai_integrated_gradients")
    logger.info("=" * 60)
    logger.info(f"Captum Integrated Gradients — Negation Paradox")
    logger.info(f"Model: {model_key} | n_steps: {n_steps}")
    logger.info("=" * 60)

    set_all_seeds(42)

    # ── Load model ─────────────────────────────────────────────────────────────
    model_id  = MODEL_IDS[model_key]
    logger.info(f"Loading {model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    label_map = {0: "negative", 1: "positive"}

    # ── Minimal pair analysis (always runs) ────────────────────────────────────
    from config import MINIMAL_PAIRS
    pair_results = run_minimal_pair_analysis(model, tokenizer, device, n_steps, logger)
    plot_minimal_pair(pair_results, MINIMAL_PAIRS, model_key)

    if args.pair_only:
        logger.info("Done (pair_only mode).")
        return

    # ── Full negation examples ─────────────────────────────────────────────────
    all_results = []
    all_rows    = []

    for i, text in enumerate(NEGATION_EXAMPLES):
        logger.info(f"\nExample {i+1}: \"{text}\"")

        res = compute_ig_attributions(
            text, model, tokenizer, device, n_steps, target_label=1
        )
        all_results.append(res)

        logger.info(
            f"  Prediction: {label_map[res['pred_label']]} ({res['pred_prob']:.2%}) | "
            f"Convergence delta: {res['delta']:.4f}"
        )

        ranked = sorted(
            zip(res["tokens"], res["attributions"]),
            key=lambda x: abs(x[1]), reverse=True
        )[:6]
        logger.info("  Top tokens by |attribution|:")
        for tok, val in ranked:
            bar = "▓" * int(abs(val) / max(abs(res["attributions"])) * 20)
            logger.info(f"    {tok:<15} {val:+.4f}  {bar}")

        for tok, val in zip(res["tokens"], res["attributions"]):
            all_rows.append({
                "example_idx": i,
                "text":        text,
                "token":       tok,
                "attribution": round(float(val), 6),
                "pred_label":  label_map[res["pred_label"]],
                "pred_prob":   round(res["pred_prob"], 4),
                "model":       model_key,
            })

    # ── Save & plot ────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(TABLES_DIR / "ig_negation_attributions.csv", index=False)
    logger.info("\nAttributions saved → ig_negation_attributions.csv")

    logger.info("Generating figures…")
    plot_negation_heatmap(all_results)
    for i, res in enumerate(all_results):
        plot_single_example_bar(res, i)

    logger.info("Done.")


if __name__ == "__main__":
    main()
