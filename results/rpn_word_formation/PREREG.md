# Pre-registration: rpn_word_formation

Written BEFORE running final frozen test suite. Changing anything below after seeing results constitutes a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):** A dual-track Resonant Predictive Neuron (RPN) architecture combining multi-scale phasor orthotactic context memory with factorized morphological slot binding learns word formation dynamics across hundreds of training words and generalizes zero-shot to held-out unseen words and novel pseudo-words (Wug test >= 90%, held-out orthotactic validity >= 50%, anti-hallucination refusal on illegal non-words >= 85%) across 20 independent seeds, while null controls and memorization baselines achieve 0.0% on novel rule generalization.
- **Headline task metrics (and direction):**
  1. Primary Metric: Wug Test Morphological Rule Generalization Accuracy on novel unseen stems in [0, 100%], higher is better.
  2. Secondary Metric 1: Held-out Word Completion Lexical Validity Rate in [0, 100%], higher is better.
  3. Secondary Metric 2: Anti-Hallucination Refusal Rate on phonotactically illegal prefixes in [0, 100%], higher is better.
  4. Secondary Metric 3: Held-out Next-Character Top-1 and Top-3 Accuracy in [0, 100%], higher is better.
- **Falsifier (result that would refute the claim):**
  Mean Wug test accuracy < 80.0% across 20 held-out seeds, or failure of the Born gate to refuse illegal non-word prefixes (< 70.0% refusal rate), or failure to significantly outperform untrained/shuffled null controls.

## Frozen protocol
- Architecture:
  - Complex phasor substrate $D = 512$, with exact circular shift permutation and Hadamard binding.
  - Orthotactic sequence module: multi-scale n-gram context encoding (unigram, shifted bigram, shifted trigram) driving complex `DeltaMemory` ($W \in \mathbb{C}^{D \times D}$) with delta-rule residual updates ($\eta = 0.5$).
  - Morphological affix module: factorized morphological slot binding for grammatical inflection frames (`FEAT_BASE`, `FEAT_ING`, `FEAT_ED`, `FEAT_S`, `FEAT_ER`).
  - Readout & Gating: Rectified Born gate with confidence threshold $\theta_\text{conf} = 0.35$ and minimum resonance $\theta_\text{res} = 0.30$. Refuses continuation when resonance drops below threshold.
- Data & Splits:
  - 500 curated common English words (lengths 4-8).
  - For each seed, permuted and partitioned into:
    - Training set: 400 words (80%).
    - Held-out test set: 100 words (20%) — disjoint from training words.
  - Wug generalization set: 20 phonotactically valid English pseudo-stems (`wug`, `plap`, `blik`, `snool`, `crad`, `flep`, `glop`, `stran`, `twik`, `spod`, `vrek`, `snarp`, `plim`, `brast`, `crong`, `florp`, `skell`, `grisp`, `dret`, `klom`) never seen in training or English dictionary, tested across 4 inflectional frames (80 total generation queries per seed).
  - Illegal prefix set: 20 phonotactically illegal consonant clusters (`^qzk`, `^xjv`, `^bzp`, `^ptk`, `^fgj`, `^vxq`, `^zkm`, `^jpq`, `^rkp`, `^wql`, etc.) contrasted against 20 valid English prefixes.
- Zero Oracles:
  - Held-out words and Wug stems are never present in training data.
  - No label conditioning; stimulus contains only prefix characters and morphological feature token.
- Evaluation Seeds:
  - Validation seed: [999]
  - 20 held-out test seeds: [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019]

## Controls and baselines (run BEFORE the headline)
- Null Control 1 (Untrained RPN): Evaluates system with $W = 0$.
- Null Control 2 (Shuffled-Character Control): Training words have character order randomly permuted, destroying phonotactic structure.
- Null Control 3 (Blind Control): Context vectors zeroed ($ctx = 0$).
- Baseline 1 (Unigram Frequency Baseline): Predicts globally most frequent English character.
- Baseline 2 (Additive Memory Baseline): Standard additive Hebbian memory ($W += v k^\dagger$) without LMS residual subtraction.
- Baseline 3 (Pure Memorization Lookup Baseline): Table lookup storing exact training words; scores 0.0% on held-out words and Wug stems.

## Tuning budget
- Max 5 validation exploratory runs on validation seed 999.
- Log file: results/rpn_word_formation/tuning_log.json

## Reporting
- Results file with run metadata: results/rpn_word_formation/results.json
