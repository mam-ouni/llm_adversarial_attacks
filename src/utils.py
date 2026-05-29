"""
utils.py
--------
Shared utility functions: seeds, logging, POS tagging, semantic similarity,
figure saving. Imported by all other scripts.
"""

import os
import random
import logging
import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from nltk import pos_tag, word_tokenize

from config import (
    LOGS_DIR, FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT,
    POS_CATEGORY_MAP, FONT_SIZE_TITLE, FONT_SIZE_LABEL,
    FONT_SIZE_TICK, FONT_SIZE_LEGEND
)


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(script_name: str) -> logging.Logger:
    """Create a logger that writes to both console and a timestamped log file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_file}")
    return logger


# ─── POS tagging ──────────────────────────────────────────────────────────────

def get_pos_category(word: str, tag: str) -> str:
    """Map a single NLTK POS tag to a coarse category."""
    return POS_CATEGORY_MAP.get(tag, "other")


def tokenize_with_pos(text: str) -> list[tuple[str, str, str]]:
    """
    Returns a list of (word, nltk_tag, coarse_category) tuples.

    Example:
        >>> tokenize_with_pos("This film is absolutely terrible")
        [('This', 'DT', 'stopword'), ('film', 'NN', 'noun'),
         ('is', 'VBZ', 'verb'), ('absolutely', 'RB', 'adverb'),
         ('terrible', 'JJ', 'adjective')]
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return [(word, tag, get_pos_category(word, tag)) for word, tag in tagged]


# ─── Text utilities ───────────────────────────────────────────────────────────

def count_changed_words(original: str, perturbed: str) -> tuple[int, float]:
    """
    Returns (n_changed, pct_changed) between two texts.
    Comparison is token-wise (split on whitespace), case-insensitive.
    """
    orig_words = original.lower().split()
    pert_words = perturbed.lower().split()
    min_len    = min(len(orig_words), len(pert_words))
    n_changed  = sum(1 for a, b in zip(orig_words[:min_len], pert_words[:min_len]) if a != b)
    n_changed += abs(len(orig_words) - len(pert_words))  # account for insertions/deletions
    pct        = n_changed / max(len(orig_words), 1)
    return n_changed, round(pct, 4)


# ─── Semantic similarity ──────────────────────────────────────────────────────

_USE_MODEL = None  # lazy-loaded singleton

def compute_use_similarity(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """
    Compute cosine similarity between paired sentences using
    sentence-transformers (paraphrase-MiniLM-L6-v2, a fast USE proxy).

    Returns a 1-D array of similarity scores in [−1, 1].
    """
    global _USE_MODEL
    if _USE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _USE_MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    emb_a = _USE_MODEL.encode(texts_a, show_progress_bar=False, batch_size=64)
    emb_b = _USE_MODEL.encode(texts_b, show_progress_bar=False, batch_size=64)

    scores = np.array([
        cos_sim(a.reshape(1, -1), b.reshape(1, -1))[0][0]
        for a, b in zip(emb_a, emb_b)
    ])
    return scores


# ─── Figure utilities ─────────────────────────────────────────────────────────

def apply_thesis_style() -> None:
    """Apply a clean, publication-quality matplotlib style."""
    matplotlib.rcParams.update({
        "figure.dpi":         FIGURE_DPI,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.titlesize":     FONT_SIZE_TITLE,
        "axes.labelsize":     FONT_SIZE_LABEL,
        "xtick.labelsize":    FONT_SIZE_TICK,
        "ytick.labelsize":    FONT_SIZE_TICK,
        "legend.fontsize":    FONT_SIZE_LEGEND,
        "font.family":        "DejaVu Sans",
    })


def save_figure(fig: plt.Figure, filename: str, subfolder: str = "") -> Path:
    """
    Save a matplotlib figure to results/figures/ (or a subfolder).
    Returns the full path.
    """
    target_dir = FIGURES_DIR / subfolder if subfolder else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{filename}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path.relative_to(Path(__file__).parent.parent)}")
    return path


# ─── Statistics helpers ───────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                 ci: float = 0.95) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean.
    Returns (lower, upper) bounds.
    """
    boots = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


def run_mcnemar(clean_correct: list[bool],
                adv_correct: list[bool]) -> dict:
    """
    McNemar test comparing clean vs adversarial predictions.

    clean_correct[i] = True if model was correct on example i before attack.
    adv_correct[i]   = True if model was correct on example i after attack.

    Returns a dict with: n10 (b), n01 (c), chi2, p_value, significant.
    """
    from statsmodels.stats.contingency_tables import mcnemar as mcn

    n11 = sum(1 for c, a in zip(clean_correct, adv_correct) if     c and     a)
    n10 = sum(1 for c, a in zip(clean_correct, adv_correct) if     c and not a)  # b
    n01 = sum(1 for c, a in zip(clean_correct, adv_correct) if not c and     a)  # c
    n00 = sum(1 for c, a in zip(clean_correct, adv_correct) if not c and not a)

    table  = [[n11, n10], [n01, n00]]
    result = mcn(table, exact=False, correction=True)

    return {
        "n11": n11, "n10_b": n10, "n01_c": n01, "n00": n00,
        "chi2":        round(result.statistic, 4),
        "p_value":     round(result.pvalue, 6),
        "significant": result.pvalue < 0.05,
    }
