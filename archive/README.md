# Archive — Retired Prototypes, Refuted Engines & Non-Essential Extras

This directory holds obsolete prototypes, refuted experimental modules, and large cached/telemetry artifacts removed from the active workspace during the repository reorganization (`2026-11-24`).

## Contents

| Subdirectory | Archived Items | Reason |
| :--- | :--- | :--- |
| `legacy_js_prototype/` | `experiment/`, `package.json`, `run_data_generator.js`, `run_experiment_suite.js`, `server.js`, `test_pong.js`, `plot_results.py` | Early Stage-0 JavaScript/browser prototype superseded by the Python `rpn/neuron_v2/` package. |
| `refuted_crfn_maze_extras/` | `Launch_Maze_GUI.bat`, `crfn_live_terminal_maze.py`, `crfn_live_telemetry.json` (7.46 MB), `crfn_trained_cache.pkl` (244 KB) | Auxiliaries and large caches for the `B-31`/`B-32` CRFN maze runs (`REFUTED` in `docs/CLAIMS.md`). |
| `refuted_experiments/` | `crfn_symbolic_engine.py` + `crfn_b47_*` + `test_crfn_symbolic_engine.py` (`B-47 REFUTED`); `crfn_causal_engine.py` + `crfn_b48_*` + `test_causal_generalization.py` (`B-48 REFUTED`); `word_formation.py` + `rpn_word_formation_benchmark.py` + `test_rpn_word_formation.py` (failed pre-registered Born-gate refusal falsifier: `58.75% < 70%`) | Refuted/self-falsified experimental modules kept for historical traceability without cluttering `rpn/neuron_v2/`. |

## Why Certain Legacy Files Remain in Place

Per `GEMINI.md` §14, `.integrity/` and `rpn/tests/integrity/` are locked (`LOCK.sha256`) and must not be modified by agents:

1. **`results/*/`:** Retained in place as immutable experimental evidence records referenced by `docs/CLAIMS.md`.
2. **16 legacy `*result*.json` files** (including root `experiment_results.json` and `rpn/*.json`): Pinned at their exact paths by `.integrity/legacy_results.txt` and `rpn/tests/integrity/test_artifact_claims.py`.
3. **Legacy `rpn/neuron_v2/{core,crfn}/` and `rpn/benchmarks/*.py` files** listed in `.integrity/lint_baseline.json` or imported by `rpn/tests/integrity/`: Pinned at their exact paths so `python .integrity/run_gates.py` stays green without weakening or regenerating locked baselines.
