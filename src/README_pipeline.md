# Pipeline d'exécution — XAI Adversarial Attack Analysis

## Ordre d'exécution

```
cd <ton_projet_root>   # le dossier qui contient src/ et results/

python src/01_run_attacks.py             # ~2–4h (200 ex × 3 modèles × 2 attaques × 3 seeds)
python src/01_run_attacks.py --quick     # ~5 min (20 ex) pour tester que tout fonctionne

python src/02_statistical_tests.py      # <1 min (lit le CSV, calcule McNemar + stats)
python src/03_xai_shap_pos.py           # ~20–40 min (SHAP sur 100 exemples)
python src/04_xai_integrated_gradients.py  # ~5 min (5 exemples de négation)
python src/05_visualize_all_results.py  # <1 min (génère tous les graphiques)
```

## Structure des sorties

```
results/
├── tables/
│   ├── attack_raw_results.csv           # sortie de 01 — une ligne par exemple attaqué
│   ├── attack_quick_summary.csv         # résumé rapide de 01
│   ├── per_seed_results.csv             # sortie de 02 — résultats par seed
│   ├── statistical_summary.csv          # sortie de 02 — mean ± std, CI par combinaison
│   ├── mcnemar_tests.csv                # sortie de 02 — chi2, p-value
│   ├── shap_pos_scores.csv              # sortie de 03 — SHAP par token
│   ├── shap_pos_aggregated.csv          # sortie de 03 — importance % par POS
│   └── ig_negation_attributions.csv     # sortie de 04 — IG sur exemples de négation
└── figures/
    ├── accuracy_drop_by_attack.png          # figure mémoire : drop d'accuracy
    ├── asr_by_model_and_attack.png          # figure mémoire : ASR comparatif
    ├── semantic_similarity_threshold.png    # figure mémoire : USE similarity
    ├── mcnemar_significance_heatmap.png     # figure mémoire : significativité
    ├── vulnerability_gap_curve.png          # figure mémoire : courbe de vulnérabilité
    ├── combined_dashboard.png              # figure mémoire : dashboard global
    ├── shap_pos_importance.png             # figure mémoire : H1 adjective obsession
    ├── shap_token_examples.png             # figure mémoire : heatmap tokens SHAP
    ├── ig_negation_heatmap.png             # figure mémoire : H2 syntax blindness
    └── ig_negation_bar_1-5.png             # figures mémoire : IG par exemple
```

## Configurer tes modèles (important)

Édite `src/config.py`, section `MODEL_IDS`, et remplace par tes checkpoints si tu as des modèles fine-tunés locaux :

```python
MODEL_IDS = {
    "bert-tiny":  "/chemin/vers/ton/bert-tiny-sst2",   
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "bert-base":  "textattack/bert-base-uncased-SST-2",
}
```

## Mode quick test

Tous les scripts supportent `--quick` ou `--n_samples 20` pour tester rapidement :

```
python src/01_run_attacks.py --quick
python src/03_xai_shap_pos.py --n_samples 20
```
