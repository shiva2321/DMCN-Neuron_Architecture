# Claim Ledger

Single source of truth for the **evidence level** of every claim in this project (rules: `GEMINI.md` sections 9-10).
Status boards / READMEs / commit messages may not state a stronger level than this file.

**Levels (use the lowest the evidence supports):**
`IMPLEMENTED` < `OBSERVED` < `REPRODUCED` < `CONTROLLED` < `GENERALIZES`; and `REFUTED`, `UNAUDITED`.

**Rules**
- Raising a row to `REPRODUCED` or above needs: results file with run metadata, >=20 seeds, the applicable null
  controls and strong baselines (GEMINI.md sections 5 and 7) linked in *Evidence*, and named *Gate* tests that enforce it.
- Agents may only LOWER levels. Raising a level requires the human owner to re-baseline
  `.integrity/claim_levels_snapshot.json` (the gate fails otherwise).
- Never delete a `REFUTED` row; add a dated note instead.
- If a gate is `xfail(strict)` and starts passing, remove the marker AND update this row in the same change.

| ID | Level | Claim (exactly as you would state it) | Evidence (results path, seeds, controls run) | Gate (test names) |
|----|-------|---------------------------------------|----------------------------------------------|-------------------|
