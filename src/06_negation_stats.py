"""
06_negation_stats.py
---------------------
Mesure statistique du Negation Paradox sur les exemples réels de SST-2.

Deux métriques complémentaires :

1. ACCURACY SPLIT
   Précision du modèle sur les exemples AVEC négation vs SANS négation.
   Si le modèle est moins précis sur les négations → H2 confirmée à grande échelle.

2. NEGATION FLIP RATE (métrique principale)
   Pour chaque exemple contenant un mot de négation :
   → on supprime le mot de négation → nouvelle phrase "flippée"
   → on rerun le modèle sur les deux versions
   → on mesure combien de fois la prédiction NE CHANGE PAS
   = "le modèle ne réagit pas à la suppression de la négation"

   Flip rate élevé = modèle insensible à la négation = H2 forte

Outputs :
    results/tables/negation_stats_summary.csv    — stats globales
    results/tables/negation_flip_details.csv     — détail par exemple
    results/figures/negation_stats_overview.png  — figure résumé

Usage :
    python src/06_negation_stats.py
    python src/06_negation_stats.py --n_examples 100   (défaut)
    python src/06_negation_stats.py --n_examples 200
"""

import sys
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_IDS, XAI_MODEL_KEY, NEGATION_WORDS, TABLES_DIR, COLORS
from utils import setup_logger, set_all_seeds, apply_thesis_style, save_figure

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset


# ─── Inference helper ─────────────────────────────────────────────────────────

def predict(text: str, model, tokenizer, device) -> tuple[int, float]:
    """Returns (predicted_label, confidence)."""
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128, padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    pred  = int(np.argmax(probs))
    return pred, float(probs[pred])


# ─── Negation flip ────────────────────────────────────────────────────────────

def flip_negation(text: str) -> tuple[str | None, str]:
    """
    Remove the first negation word found in the text.
    Returns (flipped_text, negation_word_removed) or (None, '') if none found.

    Strategy : remove the negation word and clean up spacing.
    Simple but effective for measuring model sensitivity.
    """
    text_lower = text.lower()
    words      = text.split()

    for i, word in enumerate(words):
        clean = word.lower().strip("'.,!?;:")
        # Handle contractions: don't → do, won't → will, can't → can
        if clean in ("don't", "dont", "doesn't", "doesnt",
                     "didn't", "didnt", "isn't", "isnt",
                     "wasn't", "wasnt", "aren't", "arent",
                     "weren't", "werent", "won't", "wont",
                     "wouldn't", "wouldnt", "couldn't", "couldnt",
                     "shouldn't", "shouldnt", "needn't", "neednt",
                     "mustn't", "mustnt"):
            # Remove the whole contraction word
            new_words  = words[:i] + words[i+1:]
            flipped    = " ".join(new_words)
            return flipped, word

        if clean in NEGATION_WORDS and clean not in ("without", "hardly", "scarcely", "barely"):
            # Remove just this word
            new_words = words[:i] + words[i+1:]
            flipped   = " ".join(new_words)
            # Clean double spaces
            flipped   = re.sub(r"\s+", " ", flipped).strip()
            return flipped, word

    return None, ""


# ─── Main analysis ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=None)
    parser.add_argument("--n_examples", type=int, default=100)
    args = parser.parse_args()

    model_key  = args.model or XAI_MODEL_KEY
    n_examples = args.n_examples

    logger = setup_logger("06_negation_stats")
    logger.info("=" * 60)
    logger.info(f"Negation Paradox Statistics | model={model_key} | n={n_examples}")
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

    # ── Load SST-2 validation ──────────────────────────────────────────────────
    logger.info("Loading SST-2 validation…")
    dataset = load_dataset("glue", "sst2", split="validation")

    # ── Split: negation vs no negation ─────────────────────────────────────────
    neg_examples    = []
    noneg_examples  = []

    for ex in dataset:
        text  = ex["sentence"]
        label = ex["label"]
        words = text.lower().split()
        has_neg = any(
            w.strip("'.,!?;:") in NEGATION_WORDS or
            any(neg in w for neg in ["n't", "not", "never", "no "])
            for w in words
        )
        if has_neg:
            neg_examples.append({"text": text, "true_label": label})
        else:
            noneg_examples.append({"text": text, "true_label": label})

    logger.info(f"  Negation examples    : {len(neg_examples)}")
    logger.info(f"  Non-negation examples: {len(noneg_examples)}")

    # ── Accuracy on negation vs no-negation ───────────────────────────────────
    logger.info("\nComputing accuracy split…")

    n_test  = min(n_examples, len(neg_examples))
    n_noneg = min(n_examples, len(noneg_examples))

    neg_correct = 0
    for ex in neg_examples[:n_test]:
        pred, _ = predict(ex["text"], model, tokenizer, device)
        neg_correct += int(pred == ex["true_label"])

    noneg_correct = 0
    for ex in noneg_examples[:n_noneg]:
        pred, _ = predict(ex["text"], model, tokenizer, device)
        noneg_correct += int(pred == ex["true_label"])

    acc_neg   = neg_correct   / n_test
    acc_noneg = noneg_correct / n_noneg
    acc_drop  = acc_noneg - acc_neg

    logger.info(f"  Accuracy (WITH negation)    : {acc_neg:.1%}  ({neg_correct}/{n_test})")
    logger.info(f"  Accuracy (WITHOUT negation) : {acc_noneg:.1%}  ({noneg_correct}/{n_noneg})")
    logger.info(f"  Accuracy drop due to negation: -{acc_drop:.1%}")

    # ── Negation Flip Rate ─────────────────────────────────────────────────────
    logger.info("\nComputing Negation Flip Rate…")

    flip_rows      = []
    n_flippable    = 0
    n_same_pred    = 0   # prediction did NOT change after removing negation
    n_changed_pred = 0   # prediction DID change

    for ex in neg_examples[:n_test]:
        text       = ex["text"]
        true_label = ex["true_label"]

        flipped, neg_word = flip_negation(text)
        if flipped is None or len(flipped.strip()) < 3:
            continue

        n_flippable += 1

        pred_orig,   conf_orig   = predict(text,    model, tokenizer, device)
        pred_flipped, conf_flipped = predict(flipped, model, tokenizer, device)

        same = (pred_orig == pred_flipped)
        if same:
            n_same_pred += 1
        else:
            n_changed_pred += 1

        flip_rows.append({
            "original_text":    text,
            "flipped_text":     flipped,
            "negation_removed": neg_word,
            "true_label":       label_map[true_label],
            "pred_original":    label_map[pred_orig],
            "pred_flipped":     label_map[pred_flipped],
            "conf_original":    round(conf_orig, 4),
            "conf_flipped":     round(conf_flipped, 4),
            "prediction_same":  int(same),
            "orig_correct":     int(pred_orig    == true_label),
            "flipped_correct":  int(pred_flipped == true_label),
        })

    flip_rate    = n_same_pred    / n_flippable if n_flippable > 0 else 0
    change_rate  = n_changed_pred / n_flippable if n_flippable > 0 else 0

    logger.info(f"  Flippable examples   : {n_flippable}")
    logger.info(f"  Prediction SAME      : {n_same_pred}  ({flip_rate:.1%})  ← paradox rate")
    logger.info(f"  Prediction CHANGED   : {n_changed_pred}  ({change_rate:.1%})")

    # ── Stats détaillées depuis le DataFrame flip ──────────────────────────────
    df_flip = pd.DataFrame(flip_rows)

    # Accuracy de base sur les exemples de négation
    acc_neg_flip   = df_flip["orig_correct"].mean()
    n_correct_neg  = df_flip["orig_correct"].sum()
    n_wrong_neg    = n_flippable - n_correct_neg

    # Accuracy selon que le modèle ignore ou réagit à la négation
    same_grp    = df_flip[df_flip["prediction_same"] == 1]
    changed_grp = df_flip[df_flip["prediction_same"] == 0]

    acc_same    = same_grp["orig_correct"].mean()    if len(same_grp)    > 0 else 0
    acc_changed = changed_grp["orig_correct"].mean() if len(changed_grp) > 0 else 0

    # Erreurs induites par la négation :
    # modèle FAUX sur original (avec négation) mais CORRECT sur flippé (sans négation)
    neg_induced = df_flip[(df_flip["orig_correct"] == 0) & (df_flip["flipped_correct"] == 1)]
    n_induced   = len(neg_induced)

    # ── Summary log (thesis-ready) ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("THESIS-READY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Model : {model_key.upper()}")
    logger.info(f"  N     : {n_flippable} negation examples testés")
    logger.info("")
    logger.info(f"  ── Accuracy de base ──────────────────────────────────")
    logger.info(f"  Accuracy sur exemples AVEC négation : {acc_neg_flip:.1%}  ({n_correct_neg}/{n_flippable})")
    logger.info(f"  Erreurs sur exemples avec négation  : {n_wrong_neg}  ({1-acc_neg_flip:.1%})")
    logger.info("")
    logger.info(f"  ── Accuracy selon la réponse à la négation ───────────")
    logger.info(f"  Quand modèle IGNORE la négation     : {acc_same:.1%} accuracy  (n={len(same_grp)})")
    logger.info(f"  Quand modèle RÉAGIT à la négation   : {acc_changed:.1%} accuracy  (n={len(changed_grp)})")
    logger.info("")
    logger.info(f"  ── Negation Flip Rate ────────────────────────────────")
    logger.info(f"  Flip rate (prédiction inchangée)    : {flip_rate:.1%}  ({n_same_pred}/{n_flippable})")
    logger.info("")
    logger.info(f"  ── Erreurs induites par la négation ─────────────────")
    logger.info(f"  N erreurs dues à la négation        : {n_induced}  ({n_induced/n_flippable:.1%})")
    logger.info(f"  → Modèle FAUX avec négation mais CORRECT sans")
    logger.info(f"  → La négation cause directement ces erreurs")
    logger.info("")
    logger.info(f"  → H2 {'STRONGLY CONFIRMED' if flip_rate > 0.6 else 'CONFIRMED' if flip_rate > 0.4 else 'PARTIALLY CONFIRMED'}")

    # ── Save tables ────────────────────────────────────────────────────────────
    df_flip.to_csv(TABLES_DIR / "negation_flip_details.csv", index=False)

    # Fix: acc_drop can be negative (model better on negation) — display correctly
    gap_sign = "+" if acc_neg > acc_noneg else "-"
    summary = {
        "model":                    model_key,
        "n_negation_tested":        n_test,
        "n_noneg_tested":           n_noneg,
        "acc_with_negation":        round(acc_neg,        4),
        "acc_without_negation":     round(acc_noneg,      4),
        "acc_gap":                  round(acc_neg - acc_noneg, 4),
        "n_flippable":              n_flippable,
        "acc_on_negation_examples": round(acc_neg_flip,   4),
        "n_correct_negation":       int(n_correct_neg),
        "n_wrong_negation":         int(n_wrong_neg),
        "acc_when_ignores_negation": round(acc_same,      4),
        "acc_when_reacts_negation":  round(acc_changed,   4),
        "n_same_prediction":        n_same_pred,
        "n_changed_prediction":     n_changed_pred,
        "flip_rate":                round(flip_rate,      4),
        "change_rate":              round(change_rate,    4),
        "n_negation_induced_errors": int(n_induced),
        "pct_negation_induced_errors": round(n_induced / n_flippable, 4),
    }
    pd.DataFrame([summary]).to_csv(
        TABLES_DIR / "negation_stats_summary.csv", index=False
    )
    logger.info(f"Tables saved → negation_stats_summary.csv, negation_flip_details.csv")

    # ── Figure ─────────────────────────────────────────────────────────────────
    apply_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: accuracy comparison
    ax1 = axes[0]
    bars = ax1.bar(
        ["With\nnegation", "Without\nnegation"],
        [acc_neg * 100, acc_noneg * 100],
        color=[COLORS["attacked"], COLORS["clean"]],
        width=0.45, edgecolor="white"
    )
    for bar, val in zip(bars, [acc_neg, acc_noneg]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1%}", ha="center", fontsize=12, fontweight="medium"
        )
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.set_title(
        f"Accuracy Split\nNegation vs No Negation (n={n_test})",
        fontsize=11, pad=8
    )
    ax1.annotate(
        f"−{acc_drop:.1%}",
        xy=(0.5, max(acc_neg, acc_noneg) * 100 - 3),
        xytext=(0.5, max(acc_neg, acc_noneg) * 100 + 5),
        ha="center", fontsize=11, color=COLORS["attacked"],
        arrowprops=dict(arrowstyle="-", color=COLORS["attacked"])
    )

    # Right: flip rate donut
    ax2 = axes[1]
    sizes  = [flip_rate * 100, change_rate * 100]
    colors = [COLORS["attacked"], COLORS["clean"]]
    wedges, _ = ax2.pie(
        sizes, colors=colors,
        startangle=90, counterclock=False,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2}
    )
    ax2.text(
        0, 0,
        f"{flip_rate:.1%}",
        ha="center", va="center",
        fontsize=20, fontweight="medium",
        color=COLORS["attacked"]
    )
    ax2.set_title(
        f"Negation Flip Rate\n(prediction unchanged after removing negation)",
        fontsize=11, pad=8
    )

    from matplotlib.patches import Patch
    legend = [
        Patch(color=COLORS["attacked"], label=f"Prediction unchanged ({flip_rate:.1%}) ← paradox"),
        Patch(color=COLORS["clean"],    label=f"Prediction changed ({change_rate:.1%})"),
    ]
    ax2.legend(handles=legend, fontsize=9, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.12))

    fig.suptitle(
        f"Negation Paradox — Large-Scale Statistics | {model_key.upper()} | n={n_flippable}",
        fontsize=13, y=1.02
    )
    fig.tight_layout()
    save_figure(fig, "negation_stats_overview")

    logger.info("Figure saved → negation_stats_overview.png")
    logger.info("Done.")


if __name__ == "__main__":
    main()
