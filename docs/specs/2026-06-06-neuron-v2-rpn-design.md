# Neuron v2 — The Resonant Predictive Neuron (RPN)

> **Design Spec / Concept Document**
> Status: **Draft v0.1** (concept locked; per-stage implementation specs to follow)
> Author: TANZLAB / Asta, 2026-06-06
> Lineage: evolves `Resonance_X` (RXCA, Version Ω) and feeds `NSCK`
> Scope: this is the **overall concept + staged roadmap**. Each build Stage (0–5) gets its own
> spec → plan → implementation cycle. The first implementation target is **Stage 0 + Stage 1**.

---

## 1. Purpose & North Star

Design a **new kind of digital neuron** — better, more efficient, more intelligent, and with real
memory — that becomes the **single atomic primitive** for the next generation of the
phasor/resonance cognitive stack (RXCA → NSCK).

**North star (headline anchor):** *honest long-term memory at near-zero energy.* A gradient-free,
phasor-based unit that learns continuously, remembers across three timescales, refuses to
hallucinate, and runs on a CPU today while mapping cleanly to oscillator / in-memory hardware
tomorrow.

The unit is the **Resonant Predictive Neuron (RPN)**. Tiled, RPNs form a **Resonant Predictive
Column (RPC)**; tiled columns form NSCK-v2.

**One-sentence essence.** An RPN is a complex-**phasor** unit that **resonates** (multi-band
oscillatory dendrites), **remembers** (a local delta-rule memory across three timescales),
**predicts** (a free-energy soma that learns from its own surprise), and **decides when to speak**
(Born-rule firing) — learning **continuously and locally**, with no backprop, no weight matrices,
and no attention.

---

## 2. Background & Lineage

RXCA (Resonance-X, Version Ω) already demonstrates a working, gradient-free cognitive architecture:
unit complex phasor hypervectors (HVs) in ℂ^D, holographic binding/bundling (VSA/HDC), a Modern
Hopfield "snap," Rectified Born-rule anti-hallucination gating, continuous-time episodic memory
(Fractional Power Encoding), and multi-band (Gamma/Beta/Alpha) encoding — with ~3 s training, no
heavy dependencies, and 187/187 tests passing.

RXCA's five memory/learning subsystems (`PartitionedBrain`, `SemanticBrain`, `BandBrain`,
`EpisodicMemory`, associative learner) are powerful but hand-built and partly patched (e.g.
additive writes saturate buckets, requiring `max_writes` caps and `ReSC+` consolidation). **Neuron
v2 collapses these into one uniform, composable primitive** with three principled upgrades:
delta-rule writes, bounded settling dynamics, and local predictive learning.

---

## 3. Research Foundation (2025–2026)

The frontier has independently converged on RXCA's principles, which both validates the direction
and pinpoints the gaps. Key threads (see References):

- **Resonate-and-Fire (RF) neurons** — complex-valued oscillatory state; Dendritic-RF gives each
  dendrite its own frequency band; PRF reaches O(L·logL). → the biological grounding for a phasor,
  multi-band neuron.
- **Titans (test-time memory)** — neural long-term memory learned at inference, "surprise = gradient
  of loss," adaptive forgetting/decay. → validates surprise-gated writing; supplies principled decay.
- **Delta-rule / fast-weights (Gated DeltaNet-2, KDA)** — write the *residual* with decoupled
  erase/write gates; the delta rule = one step of online prediction-error descent. → the key upgrade
  that removes RXCA's saturation problem.
- **Kuramoto oscillator memory** — memories as stable phase-locked states; honeycomb topology →
  exponential capacity, no spurious memories; higher-order coupling bridges to modern Hopfield. → a
  dynamical, provably high-capacity attractor route + hardware path.
- **VSA/HDC SOTA & limits** — known limits (high D for orthogonality, static item memory); trainable
  and structured-code variants. → motivates learnable item memory + smarter writes.
- **Predictive coding / Free Energy** — local Hebbian learning by minimizing prediction error; no
  global backprop; PC naturally produces oscillations. → the local, gradient-free "intelligence" loop.
- **Phase-based binding & memory** — binding-by-synchrony / communication-through-coherence;
  theta-gamma phase code for episodic sequence memory (phase encodes *slot/position*, not strict
  order). → composition + working-memory mechanism.
- **Cortical columns (Thousand Brains)** — each column is a semi-independent sensorimotor learner
  using reference frames ("what" + "where"), columns vote; engineering-efficient (no spikes). → the
  assembly model for RPCs.
- **Hardware** — Oscillatory Neural Networks (ReRAM-coupled ring / spintronic / photonic oscillators)
  encode info in phase; In-Memory Computing (memristor/FeFET crossbars) runs VSA ops at ~826×
  energy / 30× latency vs GPU. → RPN primitives map 1:1 to real silicon.
- **Neurosymbolic reasoning** — hybrids generalize far better and ~2 orders of magnitude more
  energy-efficiently than neural-only (e.g., Tower-of-Hanoi 95% vs 34%, generalizing to unseen
  sizes). → keep reasoning neurosymbolic.

---

## 4. Core Concept — Anatomy of one RPN

```
            +----------------------------  RESONANT PREDICTIVE NEURON  ---------------------------+
 input ---> |  DENDRITES   K phasor band-oscillators (gamma/beta/alpha), Kuramoto-coupled         |
 (any HV)   |              -> resonate to matching frequencies; bind-by-synchrony                 |
            |              -> hold the transient T0 "working-memory" phase pattern                |
            |                                   |                                                 |
            |                                   v                                                 |
            |  SOMA        PREDICTIVE / free-energy core                                          |
            |              -> predicts next input; surprise = precision-weighted error            |
            |              -> BORN-RULE adaptive threshold decides emit vs. stay-silent            |
            |                                   |                                                 |
            |  MEMORY      T0 phase pattern (now) . T1 delta-rule fast-weights (sec-min)           |
            |              . T2 consolidated couplings / item-phasors (long-term)                  |
            |                                                                                      |
            |  LEARNING    local prediction-error / delta only -> gradient-free invariant holds    |
            +--------------------------------------------------------------------------------------+
   State = unit complex phasor (C).  Compose via bind / bundle / permute  +  phase synchrony.
```

Components:

- **Dendrites** — `K` complex-phasor sub-states, each tuned to a frequency band (γ/β/α), optionally
  Kuramoto-coupled. They internalize RXCA's `BandBrain` at the single-unit level and provide
  binding-by-synchrony for transient grouping.
- **Soma** — a predictive/free-energy core that maintains a generative model (predicts its next
  input via VSA `unbind`), computes precision-weighted error (surprise), and gates output through a
  Rectified Born-rule adaptive threshold (anti-hallucination).
- **Local memory** — three timescales (Section 6), written by the delta rule and gated by surprise.
- **Learning** — purely local prediction-error/delta updates; **no global backprop** (invariant).

---

## 5. Operating Loop — the `k` dial unifies ② / ③ / ④

A single configuration parameter, `k` = number of settling micro-steps, spans the whole archetype
family explored during design:

- `k = 1` → **② Resonator Memory Cell** (one-shot, write-based; RXCA-like speed).
- `k > 1` with the predictive loop closed → **③ Predictive Neuron** (intelligence).
- `k > 1` as phase relaxation → **④ Kuramoto Attractor** (dynamical memory).

**One step:**

1. **Encode** input → phasor HV; inject into dendritic bands.
2. **Settle** for `k` micro-steps (generalizes RXCA's 6-step MHN β-annealing).
3. **Predict** the expected next input from local memory (`unbind`).
4. **Surprise** = `1 − sim(prediction, actual)` (precision-weighted error).
5. **Write** the *residual* via the **delta rule** (decoupled erase/write gates), **gated by
   surprise**, with **adaptive decay**.
6. **Fire** iff Born-rule confidence clears the adaptive threshold; else stay silent.

v1 uses **fixed `k = 4`** (Decision #5); adaptive "settle-until-confidence-plateaus" arrives in
Stage 3.

---

## 6. Memory — Three Timescales

| Tier | Substrate | Written by | Lifetime | Upgrades from RXCA |
|---|---|---|---|---|
| **T0 Working** | transient phase pattern (θ-slot, γ-item) | resonance (free) | sub-second | new |
| **T1 Episodic** | local fast-weights | delta-rule, surprise-gated, decaying | sec–min | `EpisodicMemory` + assoc. learner, de-saturated |
| **T2 Semantic** | item-phasors / couplings | consolidation / replay | long | `ItemMemory` + `ReSC+`, principled |

**Flow:** perceive → T0 phase pattern → surprise gates T1 delta-writes → adaptive decay forgets the
unremarkable → consolidation (replay scheduled by "model-time," Ebbinghaus-style) promotes
persistent T1 into T2 → recall reads T2/T1 and reinstates T0. This is hippocampal→neocortical
systems consolidation, and it turns RXCA's batch pipeline into a **continual-learning machine** with
no train/test split.

**Note on phase coding:** θ-phase encodes an item's *slot/position*, not its strict firing order
(per 2025 evidence); therefore phase is used as a frame-of-reference/separator and is paired with
FPE for true temporal order.

---

## 7. Binding, Reasoning & Composition

- **Binding (hybrid).** Exact **VSA `bind`** for *structural* relations (kept, invertible);
  **phase synchrony** for *transient* routing/grouping (a gradient-free attention gate). In v1,
  **synchrony only gates/routes; `bind` writes structure** (Decision #7).
- **Reasoning.** Algebraic backbone (RXCA `TripleStore` / echo-chaining / analogy) **+ dynamical
  relaxation** (multi-hop = settling to a coherent attractor = constraint satisfaction) **+ active
  inference** (planning) in later stages. Both explicit symbolic scaffolding and emergent geometry
  are used (Decision #3) — neurosymbolic.
- **Composition (RPC).** Columns = tiled RPNs that **vote** (Cortical-Messaging-Protocol style) and
  bind **"what" + "where"** in **reference frames**, enabling compositional world-models.

---

## 8. I/O & Senses

- VSA is **natively multimodal**: any modality → phasor HV → fused by `bind` in one space (no
  separate fusion network). Reuse RXCA's `HolographicImageEncoder` / `HolographicAudioEncoder`.
- An oscillatory/RF unit is a **native signal transducer** — audio / time-series feed directly into
  phase (the unit is a filter bank; no FFT preprocessing). Different modalities/features can occupy
  different frequency bands (frequency-division multiplexing).
- **v1 demo: language first** (Decision #1), reusing RXCA's corpus/benchmarks for parity; a small
  **audio/time-series probe** lands in Stage 2 to showcase the resonator's native edge. Full
  sensorimotor embodiment is deferred.

---

## 9. Learning Rule

- **Local prediction-error / delta only** (Decision #4 path) → **gradient-free invariant
  preserved** (no torch/autograd, no global backprop). The delta rule is one step of online
  regression on prediction error.
- Surprise gating + adaptive decay + consolidation = continual, self-supervised learning (no labels).
- Forward-Forward (contrastive positive/negative oscillatory phase) is noted as a research
  side-quest, not core.

---

## 10. Hardware Posture

**Software-first, hardware-aware op-discipline** (Decision #6 — a guideline, not a hard gate).
Restrict the unit to phase-only state, local coupling, bind/bundle, sparse/event reads, and
content-addressable lookup so it stays 1:1 mappable to **Oscillatory Neural Networks** (phase) and
**In-Memory Computing** crossbars (VSA ops) later — while preserving RXCA's CPU/numpy, ~3 s,
no-deps property now.

---

## 11. Resolved Design Decisions

| # | Decision | Resolution | Rationale |
|---|---|---|---|
| 1 | First demo | **Language first**, + audio/time-series probe in Stage 2 | Lowest-risk parity proof; existing tests/baselines |
| 2 | T2 substrate | **VSA item-phasors first**; Kuramoto couplings as Stage-4 option | Keep exact/invertible; defer dynamical capacity |
| 3 | Symbols vs emergent | **Both** (explicit scaffolding + emergent geometry) | RXCA's seeded-not-hardcoded ethos; neurosymbolic sweet spot |
| 4 | Training mode | **Bootstrap (batch) → continual online** | Reproducible base + headline continual capability |
| 5 | Settling `k` | **Fixed `k≈4` (v1)**; adaptive in Stage 3 | Protect speed/auditability now |
| 6 | HW op-discipline | **Design guideline, not a hard gate** | Keep ONN/IMC door open without hamstringing iteration |
| 7 | Synchrony writes structure? | **No (v1)** — gate/route only; `bind` writes | Preserve exactness/invertibility |
| 8 | Headline anchor | **Honest long-term memory at near-zero energy** (RULER + continual BWT + energy, no RXCA regression) | The unique, measurable win |

---

## 12. Mapping onto / Upgrading RXCA (nothing wasted)

| RXCA today | → RPN/RPC equivalent | Change |
|---|---|---|
| `BandBrain` (γ/β/α) | dendritic band-oscillators | internalized per-unit |
| `PartitionedBrain` (additive, `max_writes`) | T1 delta-rule fast-weights | de-saturated (delta) |
| `SemanticBrain` (soft kNN) | dynamical relaxation read + synchrony routing | settling |
| `EpisodicMemory` (FPE) | T1 time-binding + T0 θ-slots | unified |
| `MHN.snap` (6-step anneal) | bounded-settling read (`k`) → Kuramoto mode | generalized |
| Born rule gate | soma adaptive firing | kept |
| Associative learner (surprise α) | core local predictive loop | promoted |
| `ReSC+` | T1→T2 consolidation/replay | principled |
| `TripleStore` / `ChainReasoner` | reasoning over columns | + dynamical |
| `ItemMemory` | T2 semantic substrate | kept |

---

## 13. Invariants Preserved

- Gradient-free (no torch/tensorflow/autograd, no global backprop).
- Unit-magnitude phasors; `normalise()` discipline at every composite output.
- `bind` / `bundle` / `permute` algebra as the structural substrate.
- Precision policy: `complex64` storage, `complex128` accumulation, `complex64` output.
- Rectified Born-rule anti-hallucination gate.
- CPU/numpy, minimal dependencies; fast (seconds-scale) bootstrap training.

---

## 14. Staged Roadmap

Each stage gets its own spec → plan → implementation, and must meet its exit criterion before the
next begins.

| Stage | Deliverable | Exit criterion |
|---|---|---|
| **0** | Retrofit **delta-rule writes** into RXCA's brain (residual + erase/write gates) | Saturation fixed; **no regression** on RXCA's 187 tests + benches |
| **1** | **RPN core (`k=1`)**: band dendrites + local delta-memory + Born firing; reproduce RXCA behaviors as one primitive | **Pillar-0 parity** with RXCA-Ω |
| **2** | **T1/T2 memory hierarchy + consolidation/replay**; continual online learning; audio/time-series probe | **Pillar-1** wins (RULER subset, continual BWT ≥ baseline) |
| **3** | **Close the predictive loop (`k>1`)**: local prediction-error as core; adaptive settling | Intelligence + efficiency gains vs Stage 1 |
| **4** | **Dynamical (Kuramoto) mode** (optional): phase-attractor T2 | Capacity/robustness gain; hardware-op alignment validated |
| **5** | **Tile into RPC columns + reasoning**; reference frames; voting | Compositional generalization vs baselines |

---

## 15. Evaluation Scorecard

Three pillars define "better." Baselines: **RXCA-Ω (self)**, BRF/Dendritic-RF (spiking),
Mamba/Gated-DeltaNet-2 (SSM), Titans (memory), a small transformer.

| Pillar | Benchmarks | Metrics |
|---|---|---|
| **0. No regression** | RXCA's 187 tests + benches (verbatim recall, factual/multi-hop, analogy, anti-hallucination, OOD confidence gap, ~3 s train) | parity or better |
| **1. Memory + continual** | RULER (reasoning-over-context), multi-needle NIAH-2, MRCR; MMLongCL / TextLongCL; CurLL | effective-context length; OP, **BWT**, forward transfer, data-efficiency |
| **2. Efficiency** | scaling-law sweeps vs baselines | energy/op, params, train time, latency, **sparsity** (events), footprint (`D×V×8B`), energy-delay product |

**Headline anchor:** *honest long-term memory at near-zero energy* — strong RULER + continual BWT +
low energy, with the Born-gate refusal intact and **no RXCA regression**.

---

## 16. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Settling cost erodes the speed edge | Keep `k` small (fixed 4); adaptive-stop later; one-shot `k=1` fallback |
| Per-neuron memory bloat | delta-rule + adaptive decay + consolidation bound capacity; SNR accounting from RXCA AGENTS.md |
| Complexity creep | strict staging; each stage must beat the prior on the scorecard before proceeding |
| Kuramoto global-sync collapse | sparse/local coupling; tuned coupling; keep exact `bind` as backbone; Kuramoto is opt-in (Stage 4) |
| Scope overrun | resist building columns (Stage 5) before the RPN core (Stage 1) proves out |

---

## 17. Open Questions Deferred to Stage Specs

- Exact dendrite count `K` and per-band center frequencies / coupling topology.
- T1 fast-weight capacity per unit and decay schedule (tie to surprise statistics).
- Consolidation trigger policy (replay scheduling by "model-time").
- Concrete RULER/continual subset and pass/fail thresholds for Stage 2.
- Column size and Cortical-Messaging-Protocol payload for Stage 5.

---

## 18. Glossary

- **RPN** — Resonant Predictive Neuron (the v2 unit).
- **RPC** — Resonant Predictive Column (assembly of RPNs).
- **HV** — hypervector (unit complex phasor in ℂ^D).
- **bind/bundle/permute** — VSA ops: element-wise multiply / normalized sum / cyclic shift.
- **Born gate** — Rectified Born-rule probability gate (anti-hallucination).
- **FPE** — Fractional Power Encoding (continuous-time binding).
- **BWT** — backward transfer (continual-learning retention metric).
- **k** — number of settling micro-steps per read.

---

## References (research foundation)

- Resonate-and-Fire: Izhikevich (2001); Balanced RF — arXiv:2402.14603; Dendritic RF — arXiv:2509.17186.
- Titans: Learning to Memorize at Test Time — arXiv:2501.00663 (NeurIPS 2025); Google "Titans + MIRAS".
- Delta-rule / fast-weights: Gated DeltaNet-2 — arXiv:2605.22791; FG2-GDN — arXiv:2604.19021; Gated DeltaNet/Mamba-2 (ICLR 2025).
- Kuramoto memory: honeycomb — arXiv:2604.01469; higher-order — arXiv:2507.21984; scalable — arXiv:2504.03102.
- VSA/HDC: histogram-recovery codes — arXiv:2511.01838; THDC — arXiv:2602.00116.
- Predictive coding / Free Energy: survey — arXiv:2407.04117; tPC-RTRL — arXiv:2602.18131; PC-ALM — arXiv:2605.31022.
- Phase binding/memory: binding-by-synchrony (Scholarpedia); theta-gamma episodic — PMC5039104; phase-of-firing & order — Nature Neuroscience s41593-025-01893-7; oscillatory-predictive unification — arXiv:2601.04362.
- Cortical columns: Thousand-Brains Systems — arXiv:2507.04494; Thousand Brains Project — arXiv:2412.18354.
- Hardware: ONN with ReRAM ring oscillators — arXiv:2503.14126; SpinONN (IOP, 2025); IMC-HDC — Nature Sci. Rep. s41598-022-23116-w, XL-HD — arXiv:2605.24788; memristor associative memory — Nature Comms s41467-026-69958-0.
- Neurosymbolic reasoning: NeuroVSA (IBM Research); neurosymbolic concepts (CACM); compositional neuro-symbolic — arXiv:2604.02434.
- Continual learning / long-context eval: Long-CL — arXiv:2505.09952; MSSR — arXiv:2603.09892; FOREVER — arXiv:2601.03938; CurLL — ACL 2025 BabyLM; RULER / NIAH-2 / MRCR.
