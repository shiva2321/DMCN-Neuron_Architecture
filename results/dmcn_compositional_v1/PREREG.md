# Pre-registration: dmcn_compositional_v1

Written BEFORE running validation or frozen 20-seed test evaluation (GEMINI.md §4.1).

- **Claim (one sentence):** `FactorizedBilinearDMCNLayer` (unsupervised competitive sparse Oja/L1 dendritic factorization + additive & bilinear coincidence binding + local Sherman-Morrison readout, ZERO hardcoded factor indices, ZERO backprop) achieves genuine **zero-shot compositional generalization** on BOTH unseen `(Color, Shape)` attribute combinations (`>= 95.0%` zero-shot accuracy) AND unseen `(Operator, Distributed-Operand)` bilinear rule compositions (`<= 0.05` zero-shot MSE, `>= 90.0%` cosine/nearest-match accuracy), beating Dense Entangled DMCN, `ablate_bilinear`, and standard PyTorch Backprop MLPs (SGD & Adam) across 20 independent seeds `[2000..2019]` with randomly permuted input columns.

## Headline Metrics & Pre-Registered Falsifiers
- **Primary Metrics:**
  - Task 1 (Zero-Shot Attribute Composition — `4 Colors x 4 Shapes`, 12 seen / 4 NEVER-SEEN pairs): Zero-shot held-out exact-pair accuracy (%) on the 4 unseen pairs.
  - Task 2 (Zero-Shot Rule x Operand Composition — `4 Operators x 12 Distributed Operands`, 36 seen / 12 NEVER-SEEN pairs): Zero-shot MSE (lower is better) and nearest-prototype accuracy (%) on the 12 unseen pairs.
- **Falsifiers (Claim Refuted If):**
  1. Task 1 zero-shot accuracy `< 95.0%` OR Task 2 zero-shot MSE `> 0.05` over 20 seeds `[2000..2019]`, OR
  2. Fails to beat Dense Entangled DMCN AND PyTorch Backprop MLP (SGD & Adam) on Task 2 zero-shot MSE by `>= 3.0x`, OR
  3. `ablate_bilinear` matches or beats intact `FactorizedBilinearDMCNLayer` on Task 2, OR
  4. Any null control (`Untrained`, `Blind`, `Shuffled Labels`) exceeds `20.0%` accuracy.

## Frozen Protocol
- All input feature columns are randomly permuted per seed (`rng.permutation(d_in)`) so zero index/slot oracle is possible.
- Disjoint zero-shot holdout: test combinations are NEVER seen during unsupervised Oja discovery OR supervised learning.
- Validation seed: `[999]` (`results/dmcn_compositional_v1/tuning_log.json`, disjoint pool `[980..999]`).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (`results/dmcn_compositional_v1/results.json`).
