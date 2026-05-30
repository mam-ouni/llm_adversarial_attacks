"""
01_run_attacks.py
-----------------
Runs TextFooler and PWWS on the 3 BERT models (Tiny, DistilBERT, Base)
with 3 fixed seeds. Collects per-example results and saves them to:

    results/tables/attack_raw_results.csv

Each row = one attacked example, with:
    model, attack, seed, original_text, perturbed_text,
    true_label, original_pred, perturbed_pred,
    original_correct, attack_success,
    n_words_changed, pct_words_changed, use_similarity

This CSV is the input for 02_statistical_tests.py and 05_visualize_all_results.py.

Usage:
    python src/01_run_attacks.py
    python src/01_run_attacks.py --quick   # 20 examples only, for testing
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# ── make project root importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODEL_IDS, ATTACK_NAMES, SEEDS, N_EXAMPLES, QUICK_TEST,
    TABLES_DIR, DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT
)
from utils import set_all_seeds, setup_logger, count_changed_words, compute_use_similarity

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack import Attacker, AttackArgs, Attack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import PWWSRen2019
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR
from textattack.goal_functions import UntargetedClassification

import torch


# ─── TextFooler rebuilt without tensorflow ────────────────────────────────────

def build_textfooler_no_tf(model_wrapper):
    """
    TextFooler (Jin et al. 2019) rebuilt without the tensorflow/USE dependency.

    The original uses UniversalSentenceEncoder (requires tensorflow_hub) as a
    semantic similarity constraint. We replace it with WordEmbeddingDistance,
    which provides equivalent semantic preservation using the already-downloaded
    paragramcf embeddings. All other components are identical to the original.

    Our post-hoc semantic similarity (in utils.py) uses sentence-transformers
    and is reported separately in the results — this only affects the attack's
    internal candidate filtering.
    """
    transformation = WordSwapEmbedding(max_candidates=50)
    constraints = [
        RepeatModification(),
        StopwordModification(),
        WordEmbeddingDistance(min_cos_sim=0.5),
        PartOfSpeech(allow_verb_noun_swap=True),
    ]
    goal_function  = UntargetedClassification(model_wrapper)
    search_method  = GreedyWordSwapWIR(wir_method="delete")
    return Attack(goal_function, constraints, transformation, search_method)


# ─── Attack recipe registry ───────────────────────────────────────────────────

ATTACK_RECIPES = {
    "textfooler": build_textfooler_no_tf,   # no tensorflow needed
    "pwws":       PWWSRen2019,              # unchanged
}


# ─── Core function ────────────────────────────────────────────────────────────

def run_single_attack(
    model_key: str,
    attack_name: str,
    seed: int,
    n_examples: int,
    logger,
) -> list[dict]:
    """
    Run one attack on one model with one seed.
    Returns a list of per-example result dicts.
    """
    logger.info(f"  Loading model: {MODEL_IDS[model_key]}")

    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_IDS[model_key])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[model_key])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    logger.info(f"  Building attack: {attack_name} (seed={seed}, n={n_examples})")
    attack_recipe = ATTACK_RECIPES[attack_name]
    # custom builders are callables taking model_wrapper directly
    # TextAttack recipe classes use .build(model_wrapper)
    if hasattr(attack_recipe, "build"):
        attack = attack_recipe.build(model_wrapper)
    else:
        attack = attack_recipe(model_wrapper)

    dataset = HuggingFaceDataset(
        DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT
    )

    attack_args = AttackArgs(
        num_examples     = n_examples,
        random_seed      = seed,
        disable_stdout   = True,
        silent           = True,
    )

    attacker = Attacker(attack, dataset, attack_args)
    results  = attacker.attack_dataset()

    # ── Collect results ────────────────────────────────────────────────────────
    records = []
    originals  = []
    perturbeds = []

    for result in results:
        if not isinstance(result, (SuccessfulAttackResult, FailedAttackResult)):
            continue

        orig_text  = result.original_result.attacked_text.text
        pert_text  = result.perturbed_result.attacked_text.text
        true_label = result.original_result.ground_truth_output
        orig_pred  = result.original_result.output
        pert_pred  = result.perturbed_result.output
        success    = isinstance(result, SuccessfulAttackResult)

        n_changed, pct_changed = count_changed_words(orig_text, pert_text)

        originals.append(orig_text)
        perturbeds.append(pert_text)

        records.append({
            "model":            model_key,
            "attack":           attack_name,
            "seed":             seed,
            "original_text":    orig_text,
            "perturbed_text":   pert_text,
            "true_label":       true_label,
            "original_pred":    orig_pred,
            "perturbed_pred":   pert_pred,
            "original_correct": int(true_label == orig_pred),
            "attack_success":   int(success),
            "n_words_changed":  n_changed,
            "pct_words_changed": pct_changed,
            "use_similarity":   None,  # filled after batch encode
        })

    # ── Batch compute USE similarity ───────────────────────────────────────────
    if originals:
        logger.info(f"  Computing USE similarity for {len(originals)} examples…")
        sims = compute_use_similarity(originals, perturbeds)
        for rec, sim in zip(records, sims):
            rec["use_similarity"] = round(float(sim), 4)

    success_count = sum(r["attack_success"] for r in records)
    asr = success_count / len(records) if records else 0
    logger.info(
        f"  Done — {len(records)} examples | "
        f"ASR={asr:.1%} | "
        f"avg USE sim={sum(r['use_similarity'] for r in records)/max(len(records),1):.3f}"
    )

    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run only 20 examples (dev/debug mode)")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Override number of examples (e.g. --n_examples 5)")
    parser.add_argument("--model",  default=None,
                        help="Run only one model key (e.g. bert-base)")
    parser.add_argument("--attack", default=None,
                        help="Run only one attack (textfooler or pwws)")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Run only one seed (e.g. --seed 42)")
    args = parser.parse_args()

    if args.n_examples:
        n_examples = args.n_examples
    elif args.quick or QUICK_TEST:
        n_examples = 20
    else:
        n_examples = N_EXAMPLES

    models  = [args.model]  if args.model  else list(MODEL_IDS.keys())
    attacks = [args.attack] if args.attack else ATTACK_NAMES
    seeds   = [args.seed]   if args.seed   else SEEDS

    logger = setup_logger("01_run_attacks")
    logger.info("=" * 60)
    logger.info("XAI Adversarial Attack Audit — Attack Pipeline")
    logger.info(f"Models:   {models}")
    logger.info(f"Attacks:  {attacks}")
    logger.info(f"Seeds:    {seeds}")
    logger.info(f"Examples: {n_examples} per run")
    logger.info("=" * 60)

    all_records = []

    for model_key in models:
        for attack_name in attacks:
            for seed in seeds:
                set_all_seeds(seed)
                logger.info(
                    f"\n[{model_key.upper()} | {attack_name.upper()} | seed={seed}]"
                )
                try:
                    records = run_single_attack(
                        model_key, attack_name, seed, n_examples, logger
                    )
                    all_records.extend(records)
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    continue

    # ── Save to CSV ────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    out_path = TABLES_DIR / "attack_raw_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    logger.info(f"\n{'='*60}")
    logger.info(f"Saved {len(df)} records → {out_path}")

    if df.empty:
        logger.error("No records collected — check errors above.")
        return

    logger.info(f"Columns: {list(df.columns)}")

    # ── Quick summary ──────────────────────────────────────────────────────────
    summary = (
        df.groupby(["model", "attack"])
        .agg(
            n_examples     = ("attack_success", "count"),
            asr_mean       = ("attack_success", "mean"),
            asr_std        = ("attack_success", "std"),
            use_sim_mean   = ("use_similarity", "mean"),
            pct_words_mean = ("pct_words_changed", "mean"),
        )
        .round(4)
    )
    logger.info("\nQuick summary:\n" + summary.to_string())

    summary_path = TABLES_DIR / "attack_quick_summary.csv"
    summary.to_csv(summary_path)
    logger.info(f"Quick summary saved → {summary_path}")


if __name__ == "__main__":
    main()
