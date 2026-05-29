"""
config.py
---------
Central configuration for the XAI Adversarial Attack Analysis project.
All scripts import from here — change a value once, it propagates everywhere.
"""

from pathlib import Path

ROOT        = Path(__file__).parent.parent   
SRC_DIR     = ROOT / "src"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR  = RESULTS_DIR / "tables"
LOGS_DIR    = RESULTS_DIR / "logs"


for d in [FIGURES_DIR, TABLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


MODEL_IDS = {
    "bert-tiny":    "M-FAC/bert-tiny-finetuned-sst2",
    "distilbert":   "distilbert-base-uncased-finetuned-sst-2-english",
    "bert-base":    "textattack/bert-base-uncased-SST-2",
}

MODEL_LABELS = {
    "bert-tiny":  "BERT-Tiny\n(4.4M)",
    "distilbert": "DistilBERT\n(66M)",
    "bert-base":  "BERT-Base\n(110M)",
}


DATASET_NAME    = "glue"
DATASET_CONFIG  = "sst2"
DATASET_SPLIT   = "validation"
N_EXAMPLES      = 200  
QUICK_TEST      = False  


LABEL_MAP = {0: "negative", 1: "positive"}

ATTACK_NAMES = ["textfooler", "pwws"]


SEEDS = [42, 123, 999]


COSINE_CONSTRAINTS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]


XAI_MODEL_KEY = "bert-base"


SHAP_N_BACKGROUND = 50
SHAP_N_SAMPLES    = 100


IG_N_STEPS = 50


NEGATION_EXAMPLES = [
    "I don't think that this film is good",
    "It is not a great movie at all",
    "I wouldn't say this performance is impressive",
    "Nobody could claim this plot is interesting",
    "I don't find the acting compelling",
]


POS_CATEGORY_MAP = {
    "JJ":  "adjective", "JJR": "adjective", "JJS": "adjective",
    "VB":  "verb",      "VBD": "verb",      "VBG": "verb",
    "VBN": "verb",      "VBP": "verb",      "VBZ": "verb",
    "NN":  "noun",      "NNS": "noun",      "NNP": "noun",  "NNPS": "noun",
    "RB":  "adverb",    "RBR": "adverb",    "RBS": "adverb",
    "DT":  "stopword",  "IN":  "stopword",  "CC":  "stopword",
    "TO":  "stopword",  "PRP": "stopword",  "PRP$": "stopword",
    "WDT": "stopword",  "WP":  "stopword",
}
POS_ORDER = ["adjective", "verb", "noun", "adverb", "stopword", "other"]

FIGURE_DPI    = 150
FIGURE_FORMAT = "png"

COLORS = {
    "bert-tiny":    "#534AB7",   
    "distilbert":   "#1D9E75",   
    "bert-base":    "#D85A30",   
    "textfooler":   "#378ADD",  
    "pwws":         "#BA7517",   
    "clean":        "#639922",   
    "attacked":     "#E24B4A",   
    "threshold":    "#E24B4A",  
    "adjective":    "#7F77DD",
    "verb":         "#1D9E75",
    "noun":         "#378ADD",
    "adverb":       "#EF9F27",
    "stopword":     "#888780",
    "other":        "#B4B2A9",
}

FONT_SIZE_TITLE  = 13
FONT_SIZE_LABEL  = 11
FONT_SIZE_TICK   = 10
FONT_SIZE_LEGEND = 10
