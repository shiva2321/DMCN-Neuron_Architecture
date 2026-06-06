# Stage B-1 — Mini RPC: 3-RPN column + compositional binding

> **Path B · NEXT** — Path A (RXCA retrofit) is **pinned** on `docs/BOARD.md`.

**Goal:** Show that **column + bind** solves a task **one RPN** and **RXCA hologram** cannot.

**Scope:** `rpn/` only. NumPy. No RXCA repo edits.

---

## Task (compositional lookup)

Teach associations of the form:

```
(color, shape) -> object
```

Examples:

| color | shape | object |
|-------|-------|--------|
| red   | circle   | apple   |
| red   | square   | box     |
| blue  | circle   | ball    |
| blue  | square   | block   |

**Context HV** = `bind(color_hv, shape_hv)` (or bundle with role permute if needed).

**Baselines must fail or confuse:**

1. **Single RPN** — all 4 pairs in one `DeltaMemory` (interference on shared substructure).
2. **RXCA hologram** — single surface, same 4 pairs.
3. **Column (3 RPNs)** — e.g. route by `color` hash to specialist; or `what` / `where` / `integrator` roles.

**Pass:** column accuracy **≥ 95%** on held-out queries; best baseline **≤ 75%** on same set.

---

## Implementation sketch

| Piece | Location |
|-------|----------|
| `CompositionalColumn` | `rpn/neuron_v2/core/column.py` |
| Role permutes (optional) | `rpn/neuron_v2/substrate/phasor.py` — `permute` if missing |
| Benchmark | `rpn/benchmarks/compositional_bind.py` |
| Test | `rpn/tests/test_compositional_column.py` |

**Column design (v0):**

- RPN-A: learns `color -> object` given shape-fixed contexts (or color-only routing).
- RPN-B: learns `shape -> object` given color-fixed contexts.
- RPN-C: **integrator** — learns full role-bound composite context -> object.

**Read policy (v0):** query integrator + specialists; **if integrator Born-passes, return integrator** (composite binding is authoritative); else return highest `best_prob` among passing specialists. Strict global max-prob vote fails because role-only specialists collide when color/shape repeat across objects.

Start with **hash-route column** (already in `benchmarks/baselines.py` `RPNColumn`); extend to **bind context** and **multi-field read**.

---

## Exit criterion (B-1 done)

- [x] Benchmark script prints PASS/FAIL (`color_sharded_column` on 8×8 grid)
- [x] pytest smoke on small `D=256`
- [x] `docs/BOARD.md` card B-1 → DONE
- [ ] Decision: unpin Path A or proceed to B-2

**Outcome:** Integrator-only `CompositionalColumn` ties `SingleRPNComposite` at scale. **Path B win** = `ColorShardedColumn` (one RPN per color, shape→object inside shard).

---

## Out of scope (B-1)

- RXCA integration (Path A)
- Kuramoto, `k>1` settling, T2 consolidation
- Low-rank `W` compression
