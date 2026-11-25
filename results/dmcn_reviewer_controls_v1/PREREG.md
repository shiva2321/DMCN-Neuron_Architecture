# Pre-registration: dmcn_reviewer_controls_v1

Written BEFORE running validation (`seed 999`) or frozen 20-seed test evaluation (`[2000..2019]`).

- **Claim (one sentence):** Even when competing baselines are strengthened with DMCN's own causal RMS homeostasis (`GRU + CausalRMS + ER-200`, `Pooled ACIL/RanPAC + CausalRMS`), Unified DMCN still substantially outperforms them on streaming delayed-cue + `50x` gain-shock Split-MNIST because Pillar-2 resonant complex spines are irreplaceable; AND on standard RGB 100-class `Split-CIFAR-100` (10 sequential 10-class tasks, `0` replay, `0` task IDs) over frozen pretrained `MobileNetV2` features (`1280-D -> 4096-D`), batched CUDA DMCN matches the FP64 joint offline ridge upper bound (`< 0.1 pp` gap) while outperforming `ER-500`, `EWC`, `Adam`, and `Linear-RLS / ACIL`.
- **Headline task metrics (higher is better):**
  - `temp__dmcn_unified_acc` (%) vs `temp__gru_causal_rms_er200_acc` & `temp__pooled_acil_causal_rms_acc`
  - `c100__dmcn_acc` (%) vs `c100__joint_fp64_acc`, `c100__acil_linear_rls_acc`, `c100__mlp_er500_acc`
- **Falsifiers (Claim Refuted If Any Trigger):**
  1. `temp__dmcn_unified_acc < 88.0%` OR fails to beat BOTH `temp__pooled_acil_causal_rms_acc` and `temp__gru_causal_rms_er200_acc` by `>= 15.0 pp` over 20 seeds `[2000..2019]`.
  2. `c100__dmcn_acc < 60.0%` OR `|c100__dmcn_acc - c100__joint_fp64_acc| > 0.20 pp`.
  3. `c100__dmcn_acc` fails to beat `c100__mlp_er500_acc` AND `c100__acil_linear_rls_acc` by `>= 2.0 pp`.
  4. Any null control (`Untrained`, `Blind`, `Shuffled`, `Majority`) exceeds `15.0%` on MNIST or `5.0%` on CIFAR-100.

## Frozen protocol
- **Part A (Streaming Temporal + `50x` Gain-Shock Split-MNIST):** Exact stream from `dmcn_unified_temporal_cil_v1` (7 strips of 112px + 4 noise delay steps, gains `[1.0, 6.0, 0.2, 10.0, 0.4]` train, unseen `[0.2, 10.0]` test). Controls receive the *exact* causal RMS normalized sequence `x_norm[t]` fed to DMCN (`Pooled ACIL + CausalRMS`, `GRU + CausalRMS + Adam`, `GRU + CausalRMS + EWC`, `GRU + CausalRMS + ER-200`).
- **Part B (RGB 10-Task `Split-CIFAR-100`, 50,000 train / 10,000 test):** Downloaded via `datasets.load_dataset("uoft-cs/cifar100")`, resized to `224x224` with ImageNet normalization, embedded ONCE via frozen pretrained `MobileNetV2` (`1280-D` pooled features, zero fine-tuning on CIFAR-100). Sequential 10 tasks of 10 classes (`0..9` through `90..99`). DMCN / Block4x (`[x, tanh(W x + b), 1]`, `d_exp=4096`, `exp_scale=1.0`, `ridge=20.0`) and optimal-ridge `Linear-RLS / ACIL` (`ridge=1.0`) run 1 pass (`batch_size=256`); gradient baselines (`MLP Adam`, `EWC`, `ER-200`, `ER-500`) run `10 epochs/task` (`lr=3e-4`, replay batch `256`) so baselines are fairly trained (not under-trained on 1 epoch).
- **Seeds:** Validation = `[999]` (`tuning_log.json`, max 2 configs); Test = `[2000..2019]` (`n = 20`, run ONCE).

## Reporting
- Results written via `.integrity/results_meta.py` to `results/dmcn_reviewer_controls_v1/results.json`.
