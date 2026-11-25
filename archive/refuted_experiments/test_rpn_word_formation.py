import pytest
import numpy as np

from neuron_v2.core.word_formation import (
    RPNWordFormationSystem,
    VOCAB_500,
    WUG_STEMS,
    ILLEGAL_PREFIXES,
    LEGAL_PREFIXES,
)


def test_initialization():
    system = RPNWordFormationSystem(D=256, seed=42)
    assert system.D == 256
    assert len(system.lexicon.chars) == 29
    assert system.mem.n_writes == 0


def test_learn_and_recall_word():
    system = RPNWordFormationSystem(D=512, seed=42)
    system.learn_word("play")
    assert system.mem.n_writes == 5  # ^ -> p -> l -> a -> y -> $
    
    # Test prefix continuation
    next_c, prob, _ = system.predict_next("^pla")
    assert next_c == "y"


def test_born_refusal_on_illegal_prefix():
    system = RPNWordFormationSystem(D=512, seed=42, confidence_threshold=0.35)
    system.train_corpus(VOCAB_500[:50])
    
    # On an illegal non-word prefix never seen and impossible in English
    next_c, prob, _ = system.predict_next("^qzk")
    assert next_c is None  # Refused to hallucinate


def test_wug_generalization_on_novel_stem():
    system = RPNWordFormationSystem(D=512, seed=42)
    # Train only standard inflection paradigms
    system.train_corpus(VOCAB_500[:100])
    
    # Test Wug novel stem
    inflected_ing = system.inflect_wug("wug", "FEAT_ING")
    assert inflected_ing == "wuging"
    
    inflected_ed = system.inflect_wug("wug", "FEAT_ED")
    assert inflected_ed == "wuged"
    
    inflected_s = system.inflect_wug("wug", "FEAT_S")
    assert inflected_s == "wugs"
    
    inflected_er = system.inflect_wug("wug", "FEAT_ER")
    assert inflected_er == "wuger"


def test_held_out_generalization_eval():
    rng = np.random.RandomState(42)
    shuffled = list(rng.permutation(VOCAB_500))
    train = shuffled[:400]
    test = shuffled[400:]
    
    system = RPNWordFormationSystem(D=512, seed=42)
    system.train_corpus(train)
    
    res = system.evaluate_held_out(test)
    assert res["top1_acc"] > 12.0  # Significantly above random chance (1/29 = 3.4%)
    assert res["top3_acc"] > 25.0  # Above random chance (3/29 = 10.3%)
