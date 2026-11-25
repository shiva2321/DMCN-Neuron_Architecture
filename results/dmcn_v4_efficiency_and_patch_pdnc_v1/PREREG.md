# Pre-registration: dmcn_v4_efficiency_and_patch_pdnc_v1

Written BEFORE running `--mode val` (`seed 999`) or frozen 20-seed `--mode test` (`[2000..2019]`).

- **Claims (two targeted upgrades):**
  1. **Dual-Timescale Gram-Cholesky + 95% Sparse Ternary (`DMCN v4`):** Replacing per-batch dense Woodbury `P_prec` solves with exact normal-equation accumulation (`G_gram += X^T X`, `H_cross += X^T Y`) factored at task checkpoints via Cholesky eliminates the `4.8x` CUDA throughput gap against PyTorch Adam on Split-MNIST while matching Woodbury within `<= 0.05 pp`, and speeds up Split-CIFAR-100 (`5377-D`) by `>= 2x` — even when combined with `95%` sparse ternary `{-1, 0, +1}` dendritic encoders (`20x` fewer Layer-1 MACs).
  2. **2D Local-Patch `PDNC v2` on Raw `16x16` Pixel Pong RL:** Restricting each hidden dendritic compartment to an overlapping `4x4` spatial receptive field (`49 x 3 = 147` compartments, `4,851` active L1 synapses — `13.5x` fewer than flat `256 -> 128`) closes the `8.90 pp` Pixel Pong RL deficit against Backprop A2C without `.backward()`, without weight transport, and without weight sharing, while outperforming `Frozen Local-Patch PDNC` by `>= 25 pp`.
- **Headline Metrics:** `mnist__gram_sps`, `mnist__gram_acc`, `c100__gram_acc`, `c100__ternary_gram_acc`, `pong__patch_pdnc_hit`.
- **Falsifiers (Claim Refuted If Any Trigger):**
  1. `F1`: `mnist__gram_acc < 95.5%` OR `|mnist__gram_acc - mnist__woodbury_acc| > 0.05 pp` OR `mnist__gram_sps < 3.0 * mnist__woodbury_sps`.
  2. `F2`: `c100__gram_acc < 68.0%` OR `|c100__gram_acc - c100__woodbury_acc| > 0.05 pp` OR `c100__ternary_gram_acc < 67.5%` OR `c100__gram_time_s > 0.5 * c100__woodbury_time_s`.
  3. `F3`: `pong__patch_pdnc_hit < 74.0%` OR `pong__patch_pdnc_hit < pong__patch_frozen_hit + 25.0` OR `pong__patch_pdnc_hit < pong__patch_bp_a2c_hit - 2.0`.
  4. `F4`: Any null control fails chance bounds (`c100__shuffled_acc > 3.0%` or any Pixel Pong null `> 35.0%`).
- **Seeds:** Validation `[999]` (`max 2` configs); Held-out Test `[2000..2019]` (`n = 20`, executed ONCE).
