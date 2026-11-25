# Dendritic Multiplexed Cascade Neuron (`DMCN`) — Airtight Architecture Specification (v2)

> **Status:** Pre-Implementation Design Spec (`UNVERIFIED` — no code or benchmarks run yet)  
> **Target Package:** `rpn/neuron_v2/dmcn/` (Dual backend: NumPy reference core + PyTorch `nn.Module` CUDA-ready wrapper)  
> **Integrity Standard:** Governed by [GEMINI.md](../GEMINI.md). Every mechanism has an explicit ablation switch and pre-registered falsifier; zero external `ItemTable` codebooks, zero Python episode replay lists, zero offline future array peeking, zero task-ID oracles.

---

## 1. Design Principle: One Unified Cellular Tensor, Zero Decoration

A single `DMCNLayer(d_in, d_out, n_branches=B, n_spine_scales=K)` handles every regime — single-cell nonlinear logic (`d_out=1`), 1-shot key-value associative memory (`d_in=D, d_out=D`), multi-layer task-free continual classification (`784 -> 128 -> 10`), non-stationary chaotic forecasting (`3 -> 32 -> 3`), and delayed-cue partial-observability RL (`d_obs -> 32 -> n_actions`) — using the **exact same data path and tensor layout**:

- **Slow Structural Weights:** $W^{\text{slow}} \in \mathbb{R}^{B \times d_{\text{out}} \times d_\psi}$
- **Fast Gated-Delta Weights:** $W^{\text{fast}} \in \mathbb{R}^{B \times d_{\text{out}} \times d_\psi}$
- **Branch Context Prototypes:** $C \in \mathbb{R}^{B \times d_\psi}$ with recruitment mask $m \in \{0, 1\}^B$
- **Spine State-Space Traces:** $M \in \mathbb{R}^{K \times d_{\text{in}}}$
- **Apical Feedback Alignment Matrix (multi-layer):** $B_{\text{fb}} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$

Memory complexity is strictly **$O(B \cdot d_{\text{out}} \cdot d_\psi)$** (linear in layer width and constant $O(1)$ in stream length $T$).

---

## 2. Exact Mathematical Formulation (v2 — Post-Adversarial Review)

### Pillar 1 — Strictly Causal Online Divisive & Excitability Homeostasis
Before updating running statistics with $x(t) \in \mathbb{R}^{d_{\text{in}}}$ (preventing future leakage):
$$\tilde{x}(t) = \frac{x(t) - \bar{\mu}(t - \Delta t)}{\sqrt{\bar{\sigma}^2(t - \Delta t) + \epsilon_0}}$$
Then update causal EMA statistics:
$$\bar{\mu}(t) = (1 - \alpha_h)\bar{\mu}(t-\Delta t) + \alpha_h x(t), \qquad \bar{\sigma}^2(t) = (1 - \alpha_h)\bar{\sigma}^2(t-\Delta t) + \alpha_h (x(t) - \bar{\mu}(t))^2$$
Each output unit $i$ also maintains slow intrinsic threshold homeostasis $\theta_i(t)$:
$$\theta_i(t) = \theta_i(t-\Delta t) + \eta_{\theta}\big(\bar{e}_i(t) - e^*\big)$$
*Ablation switch:* `ablate_homeostasis=True` ($\tilde{x}(t) \equiv x(t)$, $\theta_i \equiv 0$).

### Pillar 2 — Causal Diagonal Multi-Timescale Spine ODE Bank
$K$ log-spaced decay rates $\lambda_k = \tau_{\min}^{-1} (\tau_{\max}/\tau_{\min})^{(k-1)/(K-1)}$ integrated via exact Zero-Order-Hold (ZOH) for arbitrary $\Delta t > 0$:
$$m_k(t) = e^{-\lambda_k \Delta t}\, m_k(t-\Delta t) + \big(1 - e^{-\lambda_k \Delta t}\big)\,\tilde{x}(t), \qquad k = 1,\dots,K$$
Presynaptic dendritic feature vector:
$$\psi(t) = \big[\tilde{x}(t);\; m_1(t);\; \dots;\; m_K(t)\big] \in \mathbb{R}^{d_\psi}, \quad d_\psi = (1 + K) d_{\text{in}}$$
*Ablation switch:* `ablate_spine_traces=True` ($K=0$, $\psi(t) = \tilde{x}(t)$).

### Pillar 3 — Surprise-Gated Active Dendritic Compartments & Dual-Cascade Memory

1. **Surprise-Gated Dormant Branch Recruitment (Task-Free Continual Isolation):**
   - At initialization, only branch $b=0$ is recruited ($m_0 = 1$, $m_{1..B-1} = 0$). Zero pre-seeded specialists.
   - Routing over recruited branches ($b: m_b = 1$) uses normalized context similarity:
     $$s_b(t) = \langle \hat{c}_b, \hat{\psi}(t) \rangle, \qquad p_b(t) = \frac{m_b \exp(\beta_{\text{route}} s_b(t))}{\sum_{j} m_j \exp(\beta_{\text{route}} s_j(t))}$$
     (with optional top-$k$ sparsity mask).
   - **Dormant recruitment trigger (immune to white noise):** A dormant branch $b^*$ activates ($m_{b^*} \leftarrow 1$, $c_{b^*} \leftarrow \hat{\psi}(t)$) iff a free branch exists AND **either** context novelty exceeds threshold ($\max_{b: m_b=1} s_b(t) < \theta_{\text{novel}}$) **or** sustained prediction error variance $\bar{v}_{\text{err}} > \theta_{\text{err}}$ AND directional error coherence $\bar{\rho}_t = \text{EMA}\big(\cos(\delta(t), \delta(t-1))\big) > \theta_\rho$.
   - Active branch prototypes adapt slowly via competitive Oja normalization: $c_b \leftarrow \text{norm}(c_b + \eta_c p_b (\hat{\psi}(t) - c_b))$.

2. **Causal Smooth Non-Monotonic Dendritic Conductance (Zero Vanishing Deadlock):**
   Effective branch weight $W_b^{\text{eff}}(t) = W_b^{\text{slow}}(t) + \kappa_{\text{fast}} W_b^{\text{fast}}(t)$ computes local potential $u_b(t) = W_b^{\text{eff}}(t)\,\psi(t) - \theta_{\text{dend}, b}$.
   $$\phi_{\text{dend}}(u_b) = (1 - \rho_b)\,\tanh(u_b) + \rho_b\,\Big(\sigma\big(\beta(u_b - \vartheta_{\text{on}, b})\big) - \sigma\big(\beta(u_b - \vartheta_{\text{off}, b})\big)\Big)$$
   where $\vartheta_{\text{off}, b} = \vartheta_{\text{on}, b} + \text{softplus}(\Delta_{\vartheta, b}) > \vartheta_{\text{on}, b}$ and $\rho_b \in [0, 1]$ is locally adaptable.
   - Somatic output: $I_{\text{soma}}(t) = \sum_{b=1}^B p_b(t)\,\phi_{\text{dend}}(u_b(t)) - \theta(t)$, $e(t) = \phi_{\text{soma}}(I_{\text{soma}}(t))$.
   - *Ablation switches:* `ablate_nonmonotonic=True` ($\rho_b \equiv 0$), `n_branches=1`.

3. **Branch-Isolated Gated-Delta Fast Trace + Zero-Buffer Benna-Fusi Cascade:**
   - **Fast-Weight Update (active branches only; inactive branches have $p_b \approx 0$ and suffer ZERO decay):**
     $$W_b^{\text{fast}}(t) = \big(1 - p_b(t)\,\gamma_{\text{forget}}\big)\,W_b^{\text{fast}}(t-\Delta t) + p_b(t)\,\eta_{\text{fast}}\,\big(\delta(t) \odot \phi'_{\text{dend}}(u_b)\big)\,\frac{\psi(t)^\top}{1 + \|\psi(t)\|_2^2}$$
     *(In direct associative key-value mode with linear soma/dendrite readout, $\delta(t) = v(t) - \hat{v}(t)$, recovering the exact normalized Gated-Delta rule per branch.)*
   - **Zero-Buffer Synaptic Cascade Consolidation (STC-Tagged):**
     Track local update directional consistency $\nu_b(t) \in [0, 1]$ via EMA cosine similarity of consecutive fast updates $\Delta W_b^{\text{fast}}$. Only persistent, corroborated structure transfers to $W_b^{\text{slow}}$:
     $$W_b^{\text{slow}}(t) = W_b^{\text{slow}}(t-\Delta t) + p_b(t)\,\eta_{\text{consol}}\,\nu_b(t)\,\big(W_b^{\text{fast}}(t) - \lambda_{\text{eq}}\,W_b^{\text{slow}}(t-\Delta t)\big)$$
   - *Ablation switches:* `ablate_fast_delta=True`, `ablate_cascade=True`.

### Pillar 4 — Single-Phase Apical-Basal Plasticity, Activity-Gated Kolen-Pollack & 3-Factor RL
- **Top-Down Apical Signal:** Layer $\ell$ receives $a_\ell(t) = B_\ell\,\delta_{\ell+1}(t) \cdot M(t)$, producing local burst teaching signal $\delta_\ell(t) = \phi'_{\text{soma}}(I_{\text{soma}, \ell})\odot a_\ell(t)$.
- **Activity-Gated Kolen-Pollack Alignment (Protects Dormant $W^{\text{slow}}$):**
  Apical feedback weights $B_\ell$ align locally to the active forward weights without applying global weight decay to inactive $W^{\text{slow}}$ branches:
  $$\Delta B_\ell = \eta_B\,e_\ell\,\delta_{\ell+1}^\top - \lambda_B\,\|e_\ell\|_2^2\,B_\ell$$
- **3-Factor Eligibility Traces (for Delayed-Reward RL):**
  $$E_b(t) = \gamma_{\text{elig}}\,E_b(t-\Delta t) + p_b(t)\,\big(\delta_{\text{act}}(t) \odot \phi'_{\text{dend}}(u_b)\big)\,\frac{\psi(t)^\top}{1 + \|\psi(t)\|_2^2}$$
  When scalar RPE $M(t) = r(t) - \bar{r}(t)$ arrives, $W_b^{\text{fast}} \mathrel{+}= \eta_{\text{rl}}\,M(t)\,E_b(t)$.
- *Ablation switches:* `freeze_random_feedback=True`, `shuffle_apical=True`, `ablate_eligibility=True`.

---

## 3. Planned Pre-Registered Benchmark Suite (6 Experiments, 20 Disjoint Test Seeds Each)

All experiments will write `results/<exp_id>/PREREG.md`, `tuning_log.json`, and `results.json` (via `.integrity/results_meta.py`), using disjoint validation vs. held-out test seeds `[2000..2019]`:

| Exp ID | Task & Real-World Mechanism Tested | Strong Baselines (Same Script & Budget) | Null Controls & Ablations |
| :--- | :--- | :--- | :--- |
| `dmcn_e1_causal_xor` | Single-cell online causal pulse XOR & delayed temporal XOR | Point Logistic Neuron, LIF SNN, 2-Layer Backprop MLP (ref) | Untrained, Blind, Shuffled Labels, `ablate_nonmonotonic`, `ablate_spine_traces` |
| `dmcn_e2_associative_recall` | In-stream 1-shot key-value recall + mid-stream key overwrite + distractors | Additive Hebbian, Monolithic DeltaMemory, Online GRU, Persistence | Untrained, Blind, Shuffled Keys, `ablate_fast_delta` |
| `dmcn_e3_continual_regimes` | Task-free 4-regime continual nonlinear stream (zero task-IDs; BWT & avg acc) | Online Backprop MLP (SGD & Adam), Online GRU, Fixed-FA MLP, Monolithic Delta | Untrained, Shuffled Labels, `n_branches=1`, `ablate_cascade` |
| `dmcn_e4_nonstationary_chaos` | Strictly causal chaotic forecasting (Lorenz/Mackey-Glass) under $10\times$ gain shock & irregular $\Delta t$ | Persistence, Online RLS/AR Filter, Online Backprop MLP/GRU | Blind, Decoupled Plasticity, `ablate_homeostasis`, `ablate_spine_traces` |
| `dmcn_e5_delayed_cue_tmaze` | Delayed-cue memory T-maze RL (cue at $t \in [0,4]$, silent corridor, terminal reward only; memoryless cap = $50\%$) | Constant-L, Constant-R, Random, Memoryless REINFORCE MLP | Blind Sensors, Shuffled/Decoupled Reward, Untrained, `ablate_spine_traces`, `ablate_eligibility` |
| `dmcn_e6_split_mnist_cuda` | Task-free Split-MNIST continual classification on CUDA PyTorch wrapper (zero task-IDs) | Online Backprop MLP (SGD & Adam), Fixed-FA MLP, Experience-Free EWC/L2 MLP | Untrained, Shuffled Labels, `n_branches=1`, `ablate_cascade` |
