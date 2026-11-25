# Pre-registration: dmcn_master_eval_v2

Written BEFORE running validation or held-out test seeds for v2 (GEMINI.md §2.2, §4.1). `results/dmcn_master_eval_v1/` remains untouched on record; both v1 and v2 are reported side-by-side.

- **Claim (one sentence):** Optimizing `DMCN` with (1) sparse `O(k_top)` active-branch forward/learn computation via cached `W_eff` and direct BLAS operations, (2) strictly causal input+output RMS scale homeostasis (`adapt_output_scale=True`), and (3) homeostatic fast-weight decay in 3-factor RL (`gamma_forget=0.005`) reduces per-step forward MACs by >= 3x and CPU NumPy latency by >= 3x vs v1 while fixing the v1 `E4` 10x-gain-shock chaos failure (NMSE < 0.35 and beating `ablate_homeostasis`) and improving `E5` delayed-cue RL stability (mean >= 92%) without regressing on `E1a`, `E1b`, `E2`, `E3`, or `E6a` across the same 20 disjoint held-out test seeds `[2000..2019]`.

## Pre-Registered Falsifiers (v2 Specific)
- **Compute Efficiency (`E7`):** Forward MACs on `64->64->10, B=4, top_k=1` must drop from v1's `20,008` to `< 7,000`, and CPU NumPy latency must drop from v1's `280.91 us/step` to `< 90 us/step`.
- **Chaos Homeostasis (`E4`):** Post-10x-gain-shock causal Lorenz NMSE must be `< 0.35` AND lower (better) than `ablate_homeostasis`, `Persistence`, and `NLMS`.
- **Delayed-Cue RL (`E5`):** Mean accuracy over 20 held-out test seeds must be `>= 92.0%` while all memoryless/constant/blind/shuffled controls remain near `50%`.
- **No Regression:** `E1a >= 95%`, `E1b >= 88%`, `E2 >= 95%` (overall & overwrite), `E3 >= 75%` (beating Backprop SGD by >= 20 pp), `E6a >= 91%`.

## Frozen Protocol
- Validation seed: `[999]` (logged to `results/dmcn_master_eval_v2/tuning_log.json`, disjoint validation pool `[980..999]`).
- Held-out test seeds (20 independent seeds, run ONCE): `[2000..2019]` (written to `results/dmcn_master_eval_v2/results.json` via `.integrity/results_meta.py`).
