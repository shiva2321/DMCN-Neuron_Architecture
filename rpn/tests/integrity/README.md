# Domain integrity gates (Neuron_v2)

Null-control tests for the claims in `docs/CLAIMS.md`. Rules: `GEMINI.md` sections 5-7. Generic machinery: `.integrity/`
(`controls.py` supplies `Rate`, `gate_beats_controls`, `gate_depends_on_inputs`, `gate_absent_in_null`, ...).

Run:

```bash
cd rpn && python -m pytest -q -p no:cacheprovider tests/integrity          # ~3 min (maze gates dominate)
RUN_SLOW_INTEGRITY=1 INTEGRITY_SEEDS=20 python -m pytest -q tests/integrity/test_artifact_claims.py   # opt-in: re-run the multi-maze benchmark over 20 seeds (~30 min)
cd .. && python .integrity/run_gates.py            # every gate: lint + results metadata + claims + lock + these tests
```

| File | Claim (ledger) | What it runs |
|---|---|---|
| `maze_controls.py` | B-30/31/32 | Independent reference implementation of the maze protocol (deterministic seeding, `slide` and original `bounce` physics) and the controls: untrained, constant-action, goal-beacon-only, blind sensors, shuffled teacher, open arena |
| `test_maze_null_controls.py` | B-31 | Trained net vs every control; needs its sensors; survives original physics; beats the beacon heuristic; demonstrations carry information |
| `test_substrate_emergence_controls.py` | B-30 | Emergent roles / neurogenesis must be absent in the untrained / zero-input substrate |
| `test_artifact_claims.py` | B-18/20/22/29/32 | Committed result JSONs vs the claims made about them; opt-in multi-seed replication of B-32 |
| `test_baselines_known_answers.py` | B-16 | Known-answer tests for the tapped-delay and XOR-MLP baselines |
| `test_memory_controls.py` | B-17 | Plain unsharded memory with the same key vs the "laminar column" |

## Convention: `xfail(strict=True, raises=AssertionError)`

A test marked this way documents a **refuted** claim: it is expected to fail on the intended assertion. If the underlying problem is
genuinely fixed and the test starts passing, the suite fails with *XPASS(strict)*. The required response is:

1. check the result is real (re-run, more seeds, all controls);
2. update `docs/CLAIMS.md` for the affected rows (the human owner re-baselines `.integrity/claim_levels_snapshot.json` if a level is raised);
3. remove the `xfail` marker in the same change.

Never remove a marker, loosen a threshold, change a seed, or edit a control to make a gate pass. These files are covered by
`.integrity/LOCK.sha256`; changing them requires the human owner (`INTEGRITY_HUMAN_APPROVED=1 python .integrity/run_gates.py --update-lock`).

## Gate thresholds (fixed 2026-10-16, conservative design choices)

system must beat every control by >= 20 percentage points; blind success <= 50% of sighted; original-physics success >= 80%;
task metric must improve >= 10%; an "emergent" effect must be absent (< 50% / zero growth) in the untrained system; baselines
must reach >= 99% on a known-solvable task; headline numbers need >= 20 seeds with run metadata.
