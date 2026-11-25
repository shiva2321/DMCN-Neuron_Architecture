# Pre-registration: <exp_id>

Written BEFORE any run. Changing anything below after seeing results = a NEW experiment ID (GEMINI.md section 2.2).

- **Claim (one sentence):**
- **Headline task metric (and direction):**
- **Falsifier (result that would refute the claim):**

## Frozen protocol
- Data / environment / constants:
- Inputs / sensors / features (and which are oracles):
- Reward / loss / objective:
- Training budget:
- Evaluation size and success criterion (absolute, not relative to a baseline failing):
- Seeds: train=[...]  validation=[...]  test=[...]  (disjoint; test >= 20)

## Controls and baselines (run BEFORE the headline)
- Null controls (untrained / blind / constant / heuristic / shuffled / original-dynamics ...):
- Baselines and their tuning budget (must be strong, correct, unit-tested):

## Tuning budget
- Configurations allowed on validation data:
- Log file: results/<exp_id>/tuning_log.json

## Reporting
- Results file with run metadata: results/<exp_id>/results.json (use `.integrity/results_meta.py`)
- Ledger row to update: docs/CLAIMS.md -> <ID>
