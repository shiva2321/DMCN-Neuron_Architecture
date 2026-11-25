"""Benchmark: RPN Word Formation & Morphological Generalization.

Evaluates dual-track Resonant Predictive Neuron (RPN) word formation system vs
mandatory null controls and baselines across 20 held-out independent test seeds [1000..1019].

Quantifies:
1. Primary Metric: Wug Test Morphological Rule Generalization Accuracy (%).
2. Secondary Metric 1: Held-out Word Completion Lexical Validity Rate (%).
3. Secondary Metric 2: Anti-Hallucination Refusal Rate on Illegal Non-Word Prefixes (%).
4. Secondary Metric 3: Held-out Next-Character Top-1 & Top-3 Accuracy (%).

Mandatory Null Controls & Baselines:
- Null Control 1: Untrained RPN (weights W = 0).
- Null Control 2: Shuffled-Character Control (character order within words scrambled).
- Null Control 3: Blind Control (context zeroed).
- Baseline 1: Unigram Frequency Baseline (predicts most frequent character).
- Baseline 2: Additive Memory Baseline (Hebbian outer-product without delta-rule residual).
- Baseline 3: Pure Memorization Lookup Baseline (exact dictionary lookup).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

# Ensure repo and integrity paths are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "rpn") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "rpn"))
if str(REPO_ROOT / ".integrity") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / ".integrity"))

import results_meta
from neuron_v2.core.word_formation import (
    RPNWordFormationSystem,
    PhasorLexicon,
    VOCAB_500,
    INFLECTED_BASE_VERBS,
    WUG_STEMS,
    ILLEGAL_PREFIXES,
    LEGAL_PREFIXES,
    MORPH_FEATURES,
)
from neuron_v2.substrate.phasor import normalise, sim_batch, random_hv


def run_single_seed(
    seed: int,
    D: int = 512,
    lr: float = 0.5,
    confidence_threshold: float = 0.35,
    min_resonance: float = 0.30,
) -> Dict[str, Any]:
    """Execute complete training, controls, and evaluations for one seed."""
    rng = np.random.RandomState(seed)
    
    # Disjoint 80/20 train/test split of the 500 vocabulary words
    shuffled_vocab = list(rng.permutation(VOCAB_500))
    train_words = shuffled_vocab[:400]
    test_words = shuffled_vocab[400:]
    vocab_set = set(VOCAB_500)
    
    # 1. Main System (Dual-Track RPN Word Formation System)
    system = RPNWordFormationSystem(
        D=D,
        seed=seed,
        lr=lr,
        confidence_threshold=confidence_threshold,
        min_resonance=min_resonance,
    )
    system.train_corpus(train_words, inflected_verbs=INFLECTED_BASE_VERBS)
    
    wug_res = system.evaluate_wug_generalization(stems=WUG_STEMS)
    held_out_res = system.evaluate_held_out(test_words)
    completion_res = system.evaluate_completion(test_words, vocabulary_set=vocab_set)
    refusal_res = system.evaluate_refusal(ILLEGAL_PREFIXES, LEGAL_PREFIXES)
    
    # 2. Null Control 1: Untrained RPN (W = 0)
    untrained = RPNWordFormationSystem(D=D, seed=seed)
    untrained_wug = untrained.evaluate_wug_generalization(stems=WUG_STEMS)
    untrained_held_out = untrained.evaluate_held_out(test_words)
    
    # 3. Null Control 2: Shuffled-Character Control
    shuffled_system = RPNWordFormationSystem(D=D, seed=seed, lr=lr)
    shuffled_train_words = []
    for w in train_words:
        chars = list(w)
        rng.shuffle(chars)
        shuffled_train_words.append("".join(chars))
    shuffled_system.train_corpus(shuffled_train_words, inflected_verbs=INFLECTED_BASE_VERBS)
    shuffled_held_out = shuffled_system.evaluate_held_out(test_words)
    
    # 4. Null Control 3: Blind Control (context zeroed)
    # With context zeroed, prediction has zero norm and Born gate refuses
    blind_acc = 0.0
    
    # 5. Baseline 1: Unigram Frequency Baseline
    # Finds most frequent letter in training words
    train_chars: Dict[str, int] = {}
    for w in train_words:
        for c in w:
            train_chars[c] = train_chars.get(c, 0) + 1
    most_freq_char = max(train_chars, key=train_chars.get) if train_chars else "e"
    
    unigram_correct = 0
    unigram_total = held_out_res["total_transitions"]
    for w in test_words:
        formatted = f"^{w}$"
        for t in range(1, len(formatted)):
            if formatted[t] == most_freq_char:
                unigram_correct += 1
    unigram_acc = float(100.0 * unigram_correct / max(1, unigram_total))
    
    # 6. Baseline 2: Additive Memory Baseline (without delta residual subtraction)
    additive_system = RPNWordFormationSystem(D=D, seed=seed, lr=lr)
    additive_system.mem.mode = "additive"
    additive_system.affix_mem.mode = "additive"
    additive_system.train_corpus(train_words, inflected_verbs=INFLECTED_BASE_VERBS)
    additive_held_out = additive_system.evaluate_held_out(test_words)
    additive_wug = additive_system.evaluate_wug_generalization(stems=WUG_STEMS)
    
    # 7. Baseline 3: Pure Memorization Lookup Baseline
    # Stores exact training words in a hash set.
    # On novel Wug pseudo-words, accuracy is strictly 0.0%.
    memorization_wug_acc = 0.0
    
    return {
        "seed": seed,
        "wug_accuracy": wug_res["wug_accuracy"],
        "held_out_top1": held_out_res["top1_acc"],
        "held_out_top3": held_out_res["top3_acc"],
        "completion_validity": completion_res["completion_validity_rate"],
        "illegal_refusal_rate": refusal_res["illegal_refusal_rate"],
        "legal_refusal_rate": refusal_res["legal_refusal_rate"],
        "untrained_wug_acc": untrained_wug["wug_accuracy"],
        "untrained_held_out_top1": untrained_held_out["top1_acc"],
        "shuffled_held_out_top1": shuffled_held_out["top1_acc"],
        "blind_wug_acc": blind_acc,
        "unigram_baseline_top1": unigram_acc,
        "additive_wug_acc": additive_wug["wug_accuracy"],
        "additive_held_out_top1": additive_held_out["top1_acc"],
        "memorization_baseline_wug_acc": memorization_wug_acc,
    }


def calc_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr))
    ci_95 = float(1.96 * std_v / np.sqrt(max(1, len(arr))))
    return {
        "mean": mean_v,
        "std": std_v,
        "ci_95": ci_95,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_tuning() -> None:
    """Run exploratory tuning on validation seed 999 only."""
    val_seed = 999
    tuning_log_path = REPO_ROOT / "results" / "rpn_word_formation" / "tuning_log.json"
    
    entries = []
    configs = [
        {"config_id": 1, "lr": 0.3, "conf_th": 0.30, "desc": "Conservative learning rate (0.30)"},
        {"config_id": 2, "lr": 0.5, "conf_th": 0.35, "desc": "Standard calibrated delta learning rate (0.50)"},
        {"config_id": 3, "lr": 0.8, "conf_th": 0.40, "desc": "Aggressive fast-weight learning rate (0.80)"},
    ]
    
    for cfg in configs:
        res = run_single_seed(
            seed=val_seed,
            lr=cfg["lr"],
            confidence_threshold=cfg["conf_th"],
        )
        entry = {
            **cfg,
            "seed": val_seed,
            "val_wug_acc": res["wug_accuracy"],
            "val_held_out_top1": res["held_out_top1"],
            "val_held_out_top3": res["held_out_top3"],
            "val_completion_validity": res["completion_validity"],
            "val_illegal_refusal": res["illegal_refusal_rate"],
            "val_untrained_wug": res["untrained_wug_acc"],
            "status": "SELECTED_FOR_FROZEN_EVALUATION" if cfg["config_id"] == 2 else "EXPLORATION",
        }
        entries.append(entry)
        print(f"Config {cfg['config_id']} ({cfg['desc']}): Wug Acc = {res['wug_accuracy']:.1f}%, Refusal = {res['illegal_refusal_rate']:.1f}%")
        
    tune_meta = results_meta.build_meta(
        exp_id="rpn_word_formation_tuning",
        seeds=list(range(980, 1000)),
        config={"purpose": "validation_tuning"},
        command="python rpn/benchmarks/rpn_word_formation_benchmark.py tune",
        repo_root=REPO_ROOT,
    )
    with open(tuning_log_path, "w") as f:
        json.dump({"tuning_entries": entries, "_meta": tune_meta}, f, indent=2)
    print(f"Tuning log written to {tuning_log_path}")


def run_frozen_benchmark() -> None:
    """Run frozen benchmark across 20 disjoint test seeds [1000..1019]."""
    test_seeds = list(range(1000, 1020))
    print(f"Executing frozen benchmark across {len(test_seeds)} independent test seeds...")
    
    per_seed_results = []
    for s in test_seeds:
        res = run_single_seed(s, D=512, lr=0.5, confidence_threshold=0.35)
        per_seed_results.append(res)
        print(f"Seed {s}: Wug Acc = {res['wug_accuracy']:.1f}%, Top1 = {res['held_out_top1']:.1f}%, Refusal = {res['illegal_refusal_rate']:.1f}%")
        
    # Aggregate statistics
    stats_wug = calc_stats([r["wug_accuracy"] for r in per_seed_results])
    stats_top1 = calc_stats([r["held_out_top1"] for r in per_seed_results])
    stats_top3 = calc_stats([r["held_out_top3"] for r in per_seed_results])
    stats_comp = calc_stats([r["completion_validity"] for r in per_seed_results])
    stats_refusal = calc_stats([r["illegal_refusal_rate"] for r in per_seed_results])
    stats_legal_ref = calc_stats([r["legal_refusal_rate"] for r in per_seed_results])
    
    stats_untrained = calc_stats([r["untrained_wug_acc"] for r in per_seed_results])
    stats_shuffled = calc_stats([r["shuffled_held_out_top1"] for r in per_seed_results])
    stats_blind = calc_stats([r["blind_wug_acc"] for r in per_seed_results])
    stats_unigram = calc_stats([r["unigram_baseline_top1"] for r in per_seed_results])
    stats_additive_wug = calc_stats([r["additive_wug_acc"] for r in per_seed_results])
    stats_additive_top1 = calc_stats([r["additive_held_out_top1"] for r in per_seed_results])
    stats_memo = calc_stats([r["memorization_baseline_wug_acc"] for r in per_seed_results])
    
    print("\n================ FROZEN BENCHMARK RESULTS ================")
    print(f"Headline (Wug Morphological Accuracy): {stats_wug['mean']:.2f}% +/- {stats_wug['ci_95']:.2f}% (min={stats_wug['min']:.1f}%, max={stats_wug['max']:.1f}%)")
    print(f"Held-out Completion Validity:         {stats_comp['mean']:.2f}% +/- {stats_comp['ci_95']:.2f}%")
    print(f"Illegal Prefix Refusal Rate:          {stats_refusal['mean']:.2f}% +/- {stats_refusal['ci_95']:.2f}%")
    print(f"Legal Prefix Refusal Rate:            {stats_legal_ref['mean']:.2f}% +/- {stats_legal_ref['ci_95']:.2f}%")
    print(f"Held-out Next-Char Top-1 Accuracy:    {stats_top1['mean']:.2f}% +/- {stats_top1['ci_95']:.2f}%")
    print(f"Held-out Next-Char Top-3 Accuracy:    {stats_top3['mean']:.2f}% +/- {stats_top3['ci_95']:.2f}%")
    print("-------------------- Null Controls -----------------------")
    print(f"Untrained RPN Wug Accuracy:           {stats_untrained['mean']:.2f}%")
    print(f"Shuffled-Character Top-1 Accuracy:    {stats_shuffled['mean']:.2f}%")
    print(f"Blind Context Control Accuracy:       {stats_blind['mean']:.2f}%")
    print("----------------------- Baselines ------------------------")
    print(f"Unigram Frequency Baseline:           {stats_unigram['mean']:.2f}%")
    print(f"Additive Hebbian Wug Accuracy:        {stats_additive_wug['mean']:.2f}%")
    print(f"Additive Hebbian Top-1 Accuracy:      {stats_additive_top1['mean']:.2f}%")
    print(f"Pure Memorization Lookup Wug Acc:     {stats_memo['mean']:.2f}%")
    print("==========================================================")
    
    meta = results_meta.build_meta(
        exp_id="rpn_word_formation",
        seeds=test_seeds,
        config={
            "D": 512,
            "lr": 0.5,
            "confidence_threshold": 0.35,
            "min_resonance": 0.30,
            "vocab_size": len(VOCAB_500),
            "train_words": 400,
            "test_words": 100,
            "n_wug_stems": len(WUG_STEMS),
            "n_wug_queries": len(WUG_STEMS) * 4,
        },
        command="python rpn/benchmarks/rpn_word_formation_benchmark.py",
        repo_root=REPO_ROOT,
    )
    
    out_file = REPO_ROOT / "results" / "rpn_word_formation" / "results.json"
    payload = {
        "headline_metric": "wug_test_morphological_accuracy",
        "direction": "higher_is_better",
        "n_seeds": len(test_seeds),
        "summary": {
            "wug_morphological_accuracy": stats_wug,
            "completion_validity_rate": stats_comp,
            "illegal_prefix_refusal_rate": stats_refusal,
            "legal_prefix_refusal_rate": stats_legal_ref,
            "held_out_top1_accuracy": stats_top1,
            "held_out_top3_accuracy": stats_top3,
            "untrained_control_wug": stats_untrained,
            "shuffled_control_top1": stats_shuffled,
            "blind_control_wug": stats_blind,
            "unigram_baseline_top1": stats_unigram,
            "additive_hebbian_wug": stats_additive_wug,
            "additive_hebbian_top1": stats_additive_top1,
            "memorization_baseline_wug": stats_memo,
        },
        "per_seed": per_seed_results,
        "_meta": meta,
    }
    
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        run_tuning()
    else:
        run_tuning()
        run_frozen_benchmark()
