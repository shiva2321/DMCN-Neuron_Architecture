# Stage 1 — Resonant Predictive Neuron (RPN) Core MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, self-contained `neuron_v2` Python package implementing the RPN unit at `k=1` (one-shot read), whose local memory uses the **delta rule** — and prove it beats additive bundling on write-saturation, while keeping the Born-rule anti-hallucination gate.

**Architecture:** Port RXCA's proven phasor substrate (bind/unbind/bundle/sim) into a fresh package; build a `DeltaMemory` with both an `additive` baseline and a `delta` mode (decoupled write/erase gates); wrap it in an `RPN` unit that predicts a target HV from a context HV, scores candidates with `sim_batch`, and emits via the rectified Born gate (or refuses). No gradients, no torch, no attention — pure NumPy phasor algebra.

**Tech Stack:** Python 3.11+, NumPy only (runtime). pytest for tests. (Mirrors RXCA's dependency-light, gradient-free ethos.)

**Scope note:** This plan implements **Stage 1** of `docs/specs/2026-06-06-neuron-v2-rpn-design.md` only, and folds in Stage 0's *proof* (delta beats additive on saturation) as a test — **without modifying the RXCA repo**. Deferred to later stages: multi-band dendrites `K>1`, settling `k>1`, the predictive loop (③), T0/T2 memory tiers, Kuramoto mode (④), and columns (⑤).

**Implementation root & isolation (IMPORTANT):** All paths below are under the implementation root **`E:\Neuron_v2\rpn\`** (read `E:\Neuron_v2\neuron_v2\...` as `E:\Neuron_v2\rpn\neuron_v2\...`). Run all commands (pytest, pip, benchmarks, git) from `E:\Neuron_v2\rpn`. **Git is already initialized** at the repo root `E:\Neuron_v2` on branch `feature/stage1-rpn-core` — **do NOT run `git init`** (Task 1, Step 1's `git init` is superseded). A separate JavaScript project — the "NAD-Neuron v2" astro-dendritic spiking simulator (`experiment/`, `server.js`, `package.json`, `test_pong.js`) — lives at the repo root and is **OFF-LIMITS**: never read-modify it, and **only ever `git add` explicit `rpn/`-relative paths** (never `git add -A` / `git add .`).

---

## File Structure

```
E:\Neuron_v2\
├── pyproject.toml                  # package metadata + deps (numpy); pytest config
├── README.md                       # quick start + what the MVP proves
├── neuron_v2\
│   ├── __init__.py                 # version + top-level exports
│   ├── substrate\
│   │   ├── __init__.py
│   │   └── phasor.py               # ported subset: random_hv, normalise, bind, unbind, bundle, sim, sim_batch
│   └── core\
│       ├── __init__.py
│       ├── delta_memory.py         # DeltaMemory: additive baseline + delta-rule (write/erase gates)
│       ├── born_gate.py            # rectified_born: anti-hallucination emit gate
│       ├── item_table.py           # ItemTable: token -> stable phasor HV + candidate matrix
│       └── rpn.py                  # RPN: encode -> recall -> Born-gated emit (k=1)
├── tests\
│   ├── test_phasor.py              # G1-G5 algebraic identities
│   ├── test_delta_memory.py        # de-saturation: delta beats additive under repeated writes
│   ├── test_born_gate.py           # rectify, normalise, refuse-on-no-signal
│   └── test_rpn.py                 # store->recall; refuse on unseen/noise context
└── benchmarks\
    └── capacity_curve.py           # delta vs additive: rms + recall vs epochs (the headline figure)
```

Responsibilities: `substrate/phasor.py` owns the algebra; `core/delta_memory.py` owns the write/recall surface; `core/born_gate.py` owns emit/refuse; `core/item_table.py` owns token↔HV; `core/rpn.py` composes them. One responsibility per file.

---

## Task 1: Project scaffold + phasor substrate

**Files:**
- Create: `E:\Neuron_v2\pyproject.toml`
- Create: `E:\Neuron_v2\neuron_v2\__init__.py`
- Create: `E:\Neuron_v2\neuron_v2\substrate\__init__.py`
- Create: `E:\Neuron_v2\neuron_v2\substrate\phasor.py`
- Test: `E:\Neuron_v2\tests\test_phasor.py`

- [ ] **Step 1: Initialize git + scaffold files**

Run (from `E:\Neuron_v2`):

```bash
git init
```

Create `pyproject.toml`:

```toml
[project]
name = "neuron_v2"
version = "0.1.0"
description = "Resonant Predictive Neuron (RPN) - gradient-free phasor cognitive unit"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

Create `neuron_v2\__init__.py`:

```python
"""neuron_v2 - Resonant Predictive Neuron (RPN)."""
__version__ = "0.1.0"
```

Create `neuron_v2\substrate\__init__.py` and `neuron_v2\core\__init__.py` as empty files:

```python
```

- [ ] **Step 2: Write the failing test** — `tests\test_phasor.py`

```python
import numpy as np
from neuron_v2.substrate.phasor import (
    random_hv, normalise, bind, unbind, bundle, sim, sim_batch,
)

D = 4096


def test_g1_bind_is_unit_magnitude():
    a = random_hv(D, seed=1)
    b = random_hv(D, seed=2)
    ab = bind(a, b)
    assert np.allclose(np.abs(ab), 1.0, atol=1e-5)


def test_g2_unbind_recovers_b_exactly():
    a = random_hv(D, seed=1)
    b = random_hv(D, seed=2)
    rec = unbind(bind(a, b), a)
    assert sim(rec, b) > 0.999


def test_g3_self_similarity_is_one():
    v = random_hv(D, seed=3)
    assert abs(sim(v, v) - 1.0) < 1e-4


def test_g4_random_pairs_near_orthogonal():
    a = random_hv(D, seed=4)
    b = random_hv(D, seed=5)
    assert abs(sim(a, b)) < 0.1


def test_bundle_is_similar_to_each_member():
    a = random_hv(D, seed=6)
    b = random_hv(D, seed=7)
    c = bundle(a, b)
    assert sim(c, a) > 0.3 and sim(c, b) > 0.3


def test_sim_batch_matches_sim():
    M = np.stack([random_hv(D, seed=s) for s in range(8)])
    q = random_hv(D, seed=3)
    batch = sim_batch(M, q)
    for i in range(8):
        assert abs(batch[i] - sim(M[i], q)) < 1e-4
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_phasor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neuron_v2.substrate.phasor'`

- [ ] **Step 4: Write minimal implementation** — `neuron_v2\substrate\phasor.py`

```python
"""Phasor VSA substrate (ported subset of Resonance_X/resonance_x/substrate/phasor.py).

Every symbol is a vector of unit complex numbers x in C^D, x_j = e^{i*theta_j}.
Guarantees: |bind|=1 ; unbind(bind(a,b),a)=b ; sim(v,v)=1 ; E[sim(x,y)]=0.
"""
from __future__ import annotations
import numpy as np


def random_hv(D: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.RandomState(seed) if seed is not None else np.random
    phases = rng.uniform(-np.pi, np.pi, D).astype(np.float32)
    return np.exp(1j * phases).astype(np.complex64)


def normalise(v: np.ndarray) -> np.ndarray:
    mag = np.abs(v)
    out = v.copy()
    mask = mag > 1e-12
    out[mask] /= mag[mask]
    return out.astype(np.complex64)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).astype(np.complex64)


def unbind(ab: np.ndarray, a: np.ndarray) -> np.ndarray:
    return (ab * np.conj(a)).astype(np.complex64)


def bundle(*vectors: np.ndarray, normalise_output: bool = True) -> np.ndarray:
    if not vectors:
        raise ValueError("bundle() requires at least one vector")
    s = vectors[0].astype(np.complex64).copy()
    for v in vectors[1:]:
        s += v.astype(np.complex64)
    return normalise(s) if normalise_output else s.astype(np.complex64)


def sim(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.real(np.dot(x, np.conj(y)))) / len(x)


def sim_batch(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    D = matrix.shape[1]
    mat_real = matrix.view(np.float32).reshape(len(matrix), -1)
    qry_real = query.view(np.float32)
    return (mat_real @ qry_real / D).astype(np.float32)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_phasor.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml neuron_v2/ tests/test_phasor.py
git commit -m "feat: scaffold neuron_v2 + phasor substrate with G1-G5 tests"
```

---

## Task 2: DeltaMemory — the de-saturation proof

**Files:**
- Create: `E:\Neuron_v2\neuron_v2\core\delta_memory.py`
- Test: `E:\Neuron_v2\tests\test_delta_memory.py`

Design: a single phasor superposition `surface` in C^D.
- `additive` write: `surface += bind(ctx, target)` (RXCA's current rule; saturates).
- `delta` write: `surface += w*bind(ctx, target) - b*bind(ctx, predict(ctx))`
  where `predict(ctx) = normalise(unbind(surface, ctx))` is the current readout estimate.
  With `w=b=1` this is the pure delta rule (writes only the residual `target - prediction`),
  so re-storing a known association adds ~0 → no runaway saturation. `w` and `b` are the
  decoupled write/erase gates (from Gated DeltaNet-2).

- [ ] **Step 1: Write the failing test** — `tests\test_delta_memory.py`

```python
import numpy as np
from neuron_v2.substrate.phasor import random_hv, sim
from neuron_v2.core.delta_memory import DeltaMemory

D = 8192
M = 32      # distinct associations
E = 8       # training epochs (repeated writes)


def _make_pairs(n, seed=0):
    rng = np.random.RandomState(seed)
    ctxs = [random_hv(D, seed=int(rng.randint(1, 2**31 - 1))) for _ in range(n)]
    tgts = [random_hv(D, seed=int(rng.randint(1, 2**31 - 1))) for _ in range(n)]
    return ctxs, tgts


def _train_and_score(mode):
    mem = DeltaMemory(D, mode=mode)
    ctxs, tgts = _make_pairs(M, seed=42)
    for _ in range(E):
        for c, t in zip(ctxs, tgts):
            mem.write(c, t)
    recalls = [sim(mem.recall(c), t) for c, t in zip(ctxs, tgts)]
    return float(np.mean(recalls)), mem.rms()


def test_delta_beats_additive_recall_under_repeats():
    delta_recall, _ = _train_and_score("delta")
    add_recall, _ = _train_and_score("additive")
    assert delta_recall > 0.5
    assert delta_recall > add_recall + 0.1


def test_additive_surface_saturates_delta_does_not():
    _, delta_rms = _train_and_score("delta")
    _, add_rms = _train_and_score("additive")
    # Repeated writes blow up the additive surface; delta stays bounded.
    assert add_rms > 2.0 * delta_rms


def test_delta_repeated_write_is_near_noop_after_learning():
    mem = DeltaMemory(D, mode="delta")
    c = random_hv(D, seed=11)
    t = random_hv(D, seed=12)
    mem.write(c, t)
    rms_after_first = mem.rms()
    for _ in range(20):
        mem.write(c, t)
    # surface should barely grow once the association is learned
    assert mem.rms() < 1.5 * rms_after_first
    assert sim(mem.recall(c), t) > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_delta_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neuron_v2.core.delta_memory'`

- [ ] **Step 3: Write minimal implementation** — `neuron_v2\core\delta_memory.py`

```python
"""DeltaMemory - a single phasor associative surface.

Two write rules:
  additive : surface += bind(ctx, target)                      (RXCA baseline; saturates)
  delta    : surface += w*bind(ctx, target) - b*bind(ctx, pred) (writes the residual only)

predict(ctx) = normalise(unbind(surface, ctx)) is the current readout estimate of target.
"""
from __future__ import annotations
import numpy as np

from ..substrate.phasor import bind, unbind, normalise


class DeltaMemory:
    def __init__(self, D: int, mode: str = "delta",
                 write_gate: float = 1.0, erase_gate: float = 1.0):
        if mode not in ("delta", "additive"):
            raise ValueError("mode must be 'delta' or 'additive'")
        self.D = D
        self.mode = mode
        self.w = float(write_gate)
        self.b = float(erase_gate)
        self.surface = np.zeros(D, dtype=np.complex64)
        self.n_writes = 0

    def predict(self, ctx: np.ndarray) -> np.ndarray:
        """Normalised readout estimate of the target bound to ctx."""
        return normalise(unbind(self.surface, ctx))

    def write(self, ctx: np.ndarray, target: np.ndarray) -> None:
        if self.mode == "additive":
            self.surface = (self.surface + bind(ctx, target)).astype(np.complex64)
        else:  # delta
            pred = self.predict(ctx)
            update = self.w * bind(ctx, target) - self.b * bind(ctx, pred)
            self.surface = (self.surface + update).astype(np.complex64)
        self.n_writes += 1

    def recall(self, ctx: np.ndarray) -> np.ndarray:
        return normalise(unbind(self.surface, ctx))

    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.abs(self.surface) ** 2)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_delta_memory.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add neuron_v2/core/delta_memory.py tests/test_delta_memory.py
git commit -m "feat: DeltaMemory with delta-rule writes; proves de-saturation vs additive"
```

**Implementation note (post-build):** The single-vector holographic surface in Step 3 above is **algebraically degenerate** for the delta rule on unit phasors (`bind(ctx, unbind(surface, ctx)) ≈ surface`, so subtracting the rebound erases the whole memory). The shipped implementation uses a **(D×D) matrix fast-weight state** `W` with `W += lr * outer(value − W@k̂, conj(k̂))` — canonical for delta/fast-weight learning.

---

## Task 3: Born gate + ItemTable + the RPN unit

**Files:**
- Create: `E:\Neuron_v2\neuron_v2\core\born_gate.py`
- Create: `E:\Neuron_v2\neuron_v2\core\item_table.py`
- Create: `E:\Neuron_v2\neuron_v2\core\rpn.py`
- Test: `E:\Neuron_v2\tests\test_born_gate.py`, `E:\Neuron_v2\tests\test_rpn.py`

- [ ] **Step 1: Write the failing test** — `tests\test_born_gate.py`

```python
import numpy as np
from neuron_v2.core.born_gate import rectified_born


def test_rectify_eliminates_negative_and_picks_best():
    res = np.array([-0.8, 0.1, 0.9, 0.2], dtype=np.float32)
    out = rectified_born(res, gate_threshold=0.01)
    assert out.best_idx == 2
    assert out.gate_passed is True
    assert out.best_prob > 0.5


def test_no_signal_refuses():
    res = np.array([-0.5, -0.2, -0.9], dtype=np.float32)
    out = rectified_born(res, gate_threshold=0.01)
    assert out.best_idx == -1
    assert out.gate_passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_born_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'neuron_v2.core.born_gate'`

- [ ] **Step 3: Write minimal implementation** — `neuron_v2\core\born_gate.py`

```python
"""Rectified Born Rule emit gate (anti-hallucination).

raw_i = max(0, r_i)^2 ; P_i = raw_i / sum(raw) ; emit iff best_P >= threshold.
Negative resonances -> zero probability (eliminated, not flipped). If no positive
mass exists, the gate refuses (best_idx = -1).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BornResult:
    probs: np.ndarray
    best_idx: int
    best_prob: float
    gate_passed: bool


def rectified_born(resonances: np.ndarray, gate_threshold: float = 0.01) -> BornResult:
    sims = np.asarray(resonances, dtype=np.float32)
    raw = np.maximum(sims, 0.0) ** 2
    total = float(np.sum(raw))
    if total <= 1e-14:
        n = len(sims)
        return BornResult(np.zeros(n, dtype=np.float32), -1, 0.0, False)
    probs = (raw / total).astype(np.float32)
    best_idx = int(np.argmax(raw))
    best_prob = float(probs[best_idx])
    return BornResult(probs, best_idx, best_prob, best_prob >= gate_threshold)
```

- [ ] **Step 4: Write minimal implementation** — `neuron_v2\core\item_table.py`

```python
"""ItemTable - maps tokens to stable phasor HVs and builds a candidate matrix."""
from __future__ import annotations
import numpy as np

from ..substrate.phasor import random_hv


class ItemTable:
    def __init__(self, D: int, seed: int = 42):
        self.D = D
        self.seed = seed
        self._t: dict[str, np.ndarray] = {}

    def __getitem__(self, token: str) -> np.ndarray:
        if token not in self._t:
            tok_seed = (abs(hash(token)) % (2**31 - 1)) ^ self.seed
            self._t[token] = random_hv(self.D, seed=int(tok_seed) % (2**31 - 1))
        return self._t[token]

    def matrix(self) -> tuple[list[str], np.ndarray]:
        tokens = list(self._t.keys())
        mat = np.stack([self._t[t] for t in tokens]).astype(np.complex64)
        return tokens, mat
```

- [ ] **Step 5: Write minimal implementation** — `neuron_v2\core\rpn.py`

```python
"""RPN - Resonant Predictive Neuron (k=1 one-shot core).

learn(ctx, target): write the association into local DeltaMemory.
read(ctx): recall echo -> score candidates -> Born gate -> emit token or refuse (None).
"""
from __future__ import annotations
from typing import Optional

from ..substrate.phasor import sim_batch
from .delta_memory import DeltaMemory
from .born_gate import rectified_born
from .item_table import ItemTable


class RPN:
    def __init__(self, D: int, items: ItemTable, mode: str = "delta"):
        self.D = D
        self.items = items
        self.mem = DeltaMemory(D, mode=mode)

    def learn(self, ctx_token: str, target_token: str) -> None:
        self.mem.write(self.items[ctx_token], self.items[target_token])

    def read(self, ctx_token: str, gate_threshold: float = 0.05) -> tuple[Optional[str], float]:
        echo = self.mem.recall(self.items[ctx_token])
        tokens, mat = self.items.matrix()
        res = sim_batch(mat, echo)
        out = rectified_born(res, gate_threshold=gate_threshold)
        if not out.gate_passed:
            return None, out.best_prob
        return tokens[out.best_idx], out.best_prob
```

- [ ] **Step 6: Write the failing test** — `tests\test_rpn.py`

```python
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpn import RPN

D = 8192


def test_store_and_recall():
    items = ItemTable(D, seed=7)
    # touch the vocabulary so the candidate matrix contains these tokens
    for tok in ["key", "box", "truck", "garage"]:
        _ = items[tok]
    rpn = RPN(D, items, mode="delta")
    rpn.learn("key", "box")
    rpn.learn("box", "truck")
    assert rpn.read("key")[0] == "box"
    assert rpn.read("box")[0] == "truck"


def test_refuses_on_unseen_context():
    items = ItemTable(D, seed=7)
    for tok in ["key", "box", "truck", "garage"]:
        _ = items[tok]
    rpn = RPN(D, items, mode="delta")
    rpn.learn("key", "box")
    token, prob = rpn.read("garage")   # never written as a context
    assert token is None               # anti-hallucination: refuse rather than guess
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_born_gate.py tests/test_rpn.py -v`
Expected: PASS (4 passed)

- [ ] **Step 8: Commit**

```bash
git add neuron_v2/core/born_gate.py neuron_v2/core/item_table.py neuron_v2/core/rpn.py tests/test_born_gate.py tests/test_rpn.py
git commit -m "feat: Born gate + ItemTable + RPN core (k=1) with store/recall + refusal"
```

**Implementation notes (post-build):**
- `ItemTable` uses **MD5-based token seeds** (not `hash()`, which is salted per interpreter).
- `RPN.read()` gates on **raw predict magnitude** (`‖W@k̂‖/√D`) before Born scoring; `recall()` normalisation alone cannot carry confidence.
- `rectified_born(..., min_resonance=...)` blocks the “single tiny positive flake → P=1.0” failure mode.

---

## Task 4: Capacity benchmark + README

**Files:**
- Create: `E:\Neuron_v2\benchmarks\capacity_curve.py`
- Create: `E:\Neuron_v2\README.md`

- [ ] **Step 1: Write the benchmark** — `benchmarks\capacity_curve.py`

```python
"""Headline figure: delta vs additive recall + surface RMS as repeated writes pile up."""
import numpy as np

from neuron_v2.substrate.phasor import random_hv, sim
from neuron_v2.core.delta_memory import DeltaMemory

D, M = 8192, 32


def run(mode, epochs):
    rng = np.random.RandomState(42)
    ctxs = [random_hv(D, seed=int(rng.randint(1, 2**31 - 1))) for _ in range(M)]
    tgts = [random_hv(D, seed=int(rng.randint(1, 2**31 - 1))) for _ in range(M)]
    mem = DeltaMemory(D, mode=mode)
    for _ in range(epochs):
        for c, t in zip(ctxs, tgts):
            mem.write(c, t)
    recall = float(np.mean([sim(mem.recall(c), t) for c, t in zip(ctxs, tgts)]))
    return recall, mem.rms()


if __name__ == "__main__":
    print(f"{'epochs':>6} | {'delta_recall':>12} {'delta_rms':>10} | {'add_recall':>10} {'add_rms':>10}")
    for e in [1, 2, 4, 8, 16, 32]:
        dr, drms = run("delta", e)
        ar, arms = run("additive", e)
        print(f"{e:>6} | {dr:>12.3f} {drms:>10.3f} | {ar:>10.3f} {arms:>10.3f}")
```

- [ ] **Step 2: Run the benchmark**

Run: `python -m benchmarks.capacity_curve`
Expected: a table where `delta_recall` stays high (~>0.7) and `delta_rms` stays bounded as epochs grow, while `add_recall` drops and `add_rms` climbs roughly with sqrt(epochs).

- [ ] **Step 3: Write `README.md`**

```markdown
# neuron_v2 — Resonant Predictive Neuron (RPN)

A gradient-free, phasor-based cognitive unit. See `docs/specs/2026-06-06-neuron-v2-rpn-design.md`
for the full concept; this repo is **Stage 1 (RPN core, k=1)**.

## What this MVP proves
- Phasor VSA substrate with exact bind/unbind (G1–G5 tests).
- `DeltaMemory`: delta-rule writes (store only the residual) **do not saturate** under
  repeated writes, unlike additive bundling — fixing RXCA's `max_writes`/`ReSC+` problem at the root.
- `RPN`: stores context→target, recalls by resonance, and **refuses to hallucinate** via the
  rectified Born gate when there is no signal.

## Install & test
```bash
pip install -e ".[dev]"
python -m pytest -v
python -m benchmarks.capacity_curve
```

## Not yet (later stages)
Multi-band dendrites (K>1), settling k>1, predictive loop (③), T0/T2 memory tiers,
Kuramoto mode (④), columns (⑤).
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all tests: 6 + 3 + 2 + 2 = 13 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/capacity_curve.py README.md
git commit -m "feat: capacity benchmark (delta vs additive) + README"
```

---

## Self-Review

**1. Spec coverage (Stage 1 of the design spec):**
- "RPN core (k=1): band dendrites + local delta-memory + Born firing" → Tasks 2 (delta-memory), 3 (Born + RPN). *Note:* MVP uses a single band (K=1); multi-band is explicitly deferred per the spec's staging — covered by the README "Not yet" section.
- Stage 0 proof (delta beats additive) folded in → Task 2 tests + Task 4 benchmark. ✓
- Invariants (gradient-free, phasor unit-magnitude, Born gate, NumPy-only) → Task 1 substrate + Task 3 gate. ✓
- Pillar-0 parity (no-regression) is validated against RXCA separately; this MVP establishes the primitive and its own test suite. ✓ (full RXCA-parity benchmarking is a Stage-2 concern.)

**2. Placeholder scan:** No TBD/TODO; every code/test step contains complete, runnable code and exact commands with expected output. ✓

**3. Type consistency:** `DeltaMemory(D, mode=...)`, `.write(ctx, target)`, `.recall(ctx)`, `.predict(ctx)`, `.rms()`; `rectified_born(resonances, gate_threshold)` → `BornResult(probs, best_idx, best_prob, gate_passed)`; `ItemTable.__getitem__`, `.matrix() -> (tokens, mat)`; `RPN(D, items, mode)`, `.learn(ctx, target)`, `.read(ctx) -> (token|None, prob)`. Names/signatures are consistent across all tasks and tests. ✓

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-06-06-stage1-rpn-core-mvp.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.
