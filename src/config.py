"""
config.py
---------
Central configuration for the XAI Adversarial Attack Analysis project.
All scripts import from here — change a value once, it propagates everywhere.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent   # project root (parent of src/)
SRC_DIR     = ROOT / "src"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR  = RESULTS_DIR / "tables"
LOGS_DIR    = RESULTS_DIR / "logs"

# Create directories if they don't exist
for d in [FIGURES_DIR, TABLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Models ───────────────────────────────────────────────────────────────────
# Update MODEL_IDS if you use your own fine-tuned checkpoints (local paths work too).
# Keys are short names used in result files and figures.
MODEL_IDS = {
    "bert-tiny":    "M-FAC/bert-tiny-finetuned-sst2",
    "distilbert":   "distilbert-base-uncased-finetuned-sst-2-english",
    "bert-base":    "textattack/bert-base-uncased-SST-2",
}

# Display names for figures (LaTeX-safe)
MODEL_LABELS = {
    "bert-tiny":  "BERT-Tiny\n(4.4M)",
    "distilbert": "DistilBERT\n(66M)",
    "bert-base":  "BERT-Base\n(110M)",
}

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME    = "glue"
DATASET_CONFIG  = "sst2"
DATASET_SPLIT   = "validation"
N_EXAMPLES      = 200   # per attack per model (increase to 500+ for final results)
QUICK_TEST      = False  # set True to run only 20 examples (dev/debug mode)

# SST-2 label mapping
LABEL_MAP = {0: "negative", 1: "positive"}

# ─── Attacks ──────────────────────────────────────────────────────────────────
ATTACK_NAMES = ["textfooler", "pwws"]

# Seeds for 3 independent runs (mean ± std)
SEEDS = [42, 123, 999]

# Cosine similarity constraint sweep (for vulnerability gap curve)
COSINE_CONSTRAINTS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

# ─── XAI ──────────────────────────────────────────────────────────────────────
# Model used for SHAP / IG analysis (most interpretable baseline)
XAI_MODEL_KEY = "bert-base"

# Number of background samples for SHAP
SHAP_N_BACKGROUND = 50
SHAP_N_SAMPLES    = 100

# Integrated Gradients steps
IG_N_STEPS = 50

# Negation examples for Integrated Gradients analysis (H2 — Syntax Blindness)
NEGATION_EXAMPLES = [
    "I don't think that this film is good",
    "It is not a great movie at all",
    "I wouldn't say this performance is impressive",
    "Nobody could claim this plot is interesting",
    "I don't find the acting compelling",
]

# Minimal pairs — identical except for one negation token
# Perfect controlled experiment for H2: same surface, opposite logical meaning
MINIMAL_PAIRS = [
    {
        "label":              "Single negation",
        "text":               "I don't think that this film is good",
        "expected_sentiment": "negative",   # speaker thinks film is NOT good
    },
    {
        "label":              "Double negation",
        "text":               "I don't think that this film is not good",
        "expected_sentiment": "positive",   # double neg → film IS good
    },
]

# ─── POS analysis ─────────────────────────────────────────────────────────────

# Broad categories (6) — for overview figure
POS_CATEGORY_MAP = {
    "JJ":  "adjective", "JJR": "adjective", "JJS": "adjective",
    "VB":  "verb",      "VBD": "verb",      "VBG": "verb",
    "VBN": "verb",      "VBP": "verb",      "VBZ": "verb",
    "MD":  "verb",
    "NN":  "noun",      "NNS": "noun",      "NNP": "noun",  "NNPS": "noun",
    "RB":  "adverb",    "RBR": "adverb",    "RBS": "adverb",
    "DT":  "stopword",  "IN":  "stopword",  "CC":  "stopword",
    "TO":  "stopword",  "PRP": "stopword",  "PRP$": "stopword",
    "WDT": "stopword",  "WP":  "stopword",  "EX":  "stopword",
    "CD":  "other",     "FW":  "other",     "SYM": "other",
    "LS":  "other",     "UH":  "other",     "RP":  "other",
}
POS_ORDER = ["adjective", "verb", "noun", "adverb", "stopword", "other"]

# Fine-grained POS mapping (20+ tags) — for detailed figure
FINE_POS_MAP = {
    # Adjectives — broken down by degree
    "JJ":   "adj — base form",           # good, bad, terrible
    "JJR":  "adj — comparative",         # better, worse, greater
    "JJS":  "adj — superlative",         # best, worst, greatest
    # Verbs — broken down by form
    "VB":   "verb — base form",
    "VBD":  "verb — past tense",
    "VBG":  "verb — gerund/present part",
    "VBN":  "verb — past participle",
    "VBP":  "verb — non-3rd present",
    "VBZ":  "verb — 3rd person present",
    "MD":   "verb — modal",              # could, should, would, might → H2
    # Nouns
    "NN":   "noun — singular",
    "NNS":  "noun — plural",
    "NNP":  "noun — proper singular",
    "NNPS": "noun — proper plural",
    # Adverbs
    "RB":   "adverb — base",
    "RBR":  "adverb — comparative",
    "RBS":  "adverb — superlative",
    # Function words
    "DT":   "determiner",
    "IN":   "preposition/subord. conj",
    "CC":   "coordinating conj",
    "TO":   "infinitive marker",
    "PRP":  "pronoun — personal",
    "PRP$": "pronoun — possessive",
    "WDT":  "wh-determiner",
    "WP":   "wh-pronoun",
    "WRB":  "wh-adverb",
    "EX":   "existential 'there'",
    "CD":   "cardinal number",
    "RP":   "particle",
    "FW":   "foreign word",
    "UH":   "interjection",
    "SYM":  "symbol",
    "PDT":  "predeterminer",
    "POS":  "possessive ending",
}

FINE_POS_ORDER = [
    "adj — base form", "adj — comparative", "adj — superlative",
    "adverb — base", "adverb — comparative", "adverb — superlative",
    "verb — base form", "verb — past tense", "verb — gerund/present part",
    "verb — past participle", "verb — non-3rd present", "verb — 3rd person present",
    "verb — modal",
    "noun — singular", "noun — plural", "noun — proper singular", "noun — proper plural",
    "determiner", "preposition/subord. conj", "coordinating conj",
    "infinitive marker", "pronoun — personal", "pronoun — possessive",
    "wh-determiner", "wh-pronoun", "wh-adverb",
    "cardinal number", "existential 'there'", "particle",
    "foreign word", "interjection", "symbol",
]

# Negation words — directly relevant to H2 (Syntax Blindness)
NEGATION_WORDS = {
    "not", "n't", "nt", "never", "no", "neither", "nor",
    "nobody", "nothing", "nowhere", "hardly", "scarcely",
    "barely", "without", "none", "cant", "wont", "dont",
    "doesnt", "didnt", "isnt", "wasnt", "wouldnt", "shouldnt",
    "couldnt", "mustnt", "neednt",
}

# Intensifiers — amplifiers/diminishers of sentiment
INTENSIFIER_WORDS = {
    "very", "extremely", "absolutely", "incredibly", "highly", "deeply",
    "terribly", "awfully", "remarkably", "exceptionally", "particularly",
    "especially", "really", "quite", "fairly", "rather", "pretty",
    "somewhat", "totally", "completely", "utterly", "thoroughly",
    "genuinely", "truly", "honestly", "seriously", "literally",
    "insanely", "ridiculously", "unbelievably", "surprisingly",
    "disappointingly", "refreshingly", "painfully", "wonderfully",
}

# ─── Figure style ─────────────────────────────────────────────────────────────
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"

# Color palette (accessible, print-friendly)
COLORS = {
    # Models
    "bert-tiny":    "#534AB7",
    "distilbert":   "#1D9E75",
    "bert-base":    "#D85A30",
    # Attacks
    "textfooler":   "#378ADD",
    "pwws":         "#BA7517",
    # Result states
    "clean":        "#639922",
    "attacked":     "#E24B4A",
    "threshold":    "#E24B4A",
    # Broad POS
    "adjective":    "#7F77DD",
    "verb":         "#1D9E75",
    "noun":         "#378ADD",
    "adverb":       "#EF9F27",
    "stopword":     "#888780",
    "other":        "#B4B2A9",
    # Functional semantic categories
    "negation":     "#E24B4A",
    "intensifier":  "#EF9F27",
    "modal":        "#D85A30",
    "sentiment_pos": "#1D9E75",
    "sentiment_neg": "#E24B4A",
    "sentiment_neu": "#888780",
}

FONT_SIZE_TITLE  = 13
FONT_SIZE_LABEL  = 11
FONT_SIZE_TICK   = 10
FONT_SIZE_LEGEND = 10
