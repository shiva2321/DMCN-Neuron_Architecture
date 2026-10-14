# Foundational Compendium of Neuroscience, Biophysics, Cognitive Psychology, and Bio-Inspired Neural Architectures

> **Document Class:** Comprehensive Theoretical, Empirical & Engineering Reference Manual  
> **Source Base:** Synthesis of Foundational Literature + Cutting-Edge Discoveries (2020–2026) across Neurobiology, Electrophysiology, Systems Neuroscience, Cognitive Psychology, and Neuromorphic AI  
> **Target Application:** Designing, Verifying, and Publishing a True Biological Neuron and Network-Scale Cognitive Unit (`Neuron_v2` / RPN / RPC / Brain Architecture)  
> **Operational Status:** Persisted Directly to Disk (Checkpoint-Proof against System Restarts and Power Disruptions)

---

## Executive Summary & Architectural Axioms
1. **The Single Neuron is Not a Point Processor:** Pyramidal neurons in mammalian and human cortex are multi-layered computational units. Dendritic compartments compute non-linear logical functions (including XOR) locally through active voltage plateaus and calcium action potentials (Chavlis & Poirazi, 2026; Gidon et al., 2020; Shapira & Segev, 2025).
2. **Learning Without Backpropagation Requires Multiple Timescales:** The brain does not use end-to-end backpropagation through time. It operates via three coupled memory timescales:
   - **T0 Working Phase / Sub-second:** Phase-locked oscillatory resonance (Kuramoto dynamics, theta-gamma phase-amplitude coupling).
   - **T1 Episodic Fast-Weights / Seconds-to-Minutes:** Surprise-gated residual learning (Gated Delta-Rule, Three-Factor R-STDP, neuromodulatory eligibility traces).
   - **T2 Semantic Consolidation / Long-term:** Systems-level consolidation via replay (Sharp-Wave Ripples transferring fast hippocampal representations into slow cortical manifolds without catastrophic interference).
3. **Columns Bind "What" and "Where" in Metric Reference Frames:** Neocortex does not build flat pattern-matchers. Cortical columns factorize structural metric spaces (entorhinal grid cell basis / reference frames) from sensory particulars (hippocampal place cells / object features), enabling zero-shot relational generalization (Tolman-Eichenbaum Machine; Hawkins' Thousand Brains).
4. **Perception and Action are Unified Under Free Energy Minimization:** Cortical microcircuits implement predictive processing where superficial layers (L2/3) broadcast precision-weighted prediction errors in the gamma band ($30-80\,\text{Hz}$), and deep layers (L5/6) generate top-down hypotheses in alpha/beta bands ($8-30\,\text{Hz}$) (Bastos, Friston, Larkum). Active inference minimizes expected free energy by balancing epistemic curiosity (information gain) with pragmatic utility (goal attainment).

---

## Table of Contents
1. [Operational Resilience & Interruption-Proof Workflow](#1-operational-resilience--interruption-proof-workflow)
2. [Module I: Biophysical & Molecular Foundations of the Biological Neuron](#2-module-i-biophysical--molecular-foundations-of-the-biological-neuron)
   - 2.1 Membrane Biophysics, Resting Potentials & Passive Cable Theory
   - 2.2 Ion Channels: Kinetics, Selectivity Filters & Channelopathies
   - 2.3 The Action Potential, Axon Initial Segment (AIS) & Active Backpropagation (bAP)
   - 2.4 Dendritic Computation: Compartmentalization, Plateaus & Human dCaAPs (Gidon, Larkum, Poirazi)
   - 2.5 Synaptic Physiology: Quantal Neurotransmission & Vesicle Dynamics
   - 2.6 The Tripartite Synapse & Glial Biology (Astrocytes, Microglia, Oligodendrocytes)
3. [Module II: Synaptic Plasticity, Metaplasticity & Neuromodulation](#3-module-ii-synaptic-plasticity-metaplasticity--neuromodulation)
   - 3.1 Hebbian Learning & Spike-Timing-Dependent Plasticity (STDP)
   - 3.2 Burst-Dependent Plasticity & Somato-Dendritic Decoupling
   - 3.3 Three-Factor Learning Rules & Neuromodulatory Systems (DA, ACh, NE, 5-HT)
   - 3.4 Homeostatic Plasticity, BCM Sliding Thresholds & Synaptic Scaling
   - 3.5 Structural Plasticity: Spine Remodeling, Pruning & Silent Synapses
4. [Module III: Neuroanatomy, Laminar Microcircuits & Cortical Columns](#4-module-iii-neuroanatomy-laminar-microcircuits--cortical-columns)
   - 4.1 Neocortical Laminar Cytoarchitecture (Layers 1 through 6)
   - 4.2 Canonical Microcircuits: Mountcastle, Szentágothai, Douglas & Martin
   - 4.3 Predictive Coding Microcircuits: Bastos, Friston & Larkum's Apical Amplification
   - 4.4 The Cortical Column & Thousand Brains Framework (Hawkins / Numenta)
   - 4.5 Subcortical Loops: Basal Ganglia Action Selection & Thalamocortical Gating
   - 4.6 The Cerebellar Microcircuit: Marr-Albus-Ito Forward/Inverse Internal Models
5. [Module IV: Systems Neuroscience, Dynamical Coding & Brain Rhythms](#5-module-iv-systems-neuroscience-dynamical-coding--brain-rhythms)
   - 5.1 Electrophysiological Rhythms & Spectral Bands ($\delta, \theta, \alpha, \beta, \gamma$)
   - 5.2 Synchrony, Phase-Locking & Binding-by-Synchrony (Singer & Fries)
   - 5.3 Dynamical Systems: Kuramoto Coupled Oscillators & Continuous Attractor Networks (CANNs)
   - 5.4 Spatial Navigation & Metric Cognitive Maps: Place, Grid, Head-Direction & Border Cells
   - 5.5 Phase Precession, Theta-Gamma Phase-Amplitude Coupling & Working Memory Slots
   - 5.6 Memory Consolidation, Sharp-Wave Ripples & Complementary Learning Systems (CLS)
   - 5.7 Continuous Temporal Memory: Hippocampal Time Cells & State-Space History Projection (LMU)
6. [Module V: Psychology, Cognitive Science & Foundational Principles of Mind](#6-module-v-psychology-cognitive-science--foundational-principles-of-mind)
   - 6.1 Classical & Operant Conditioning: Pavlov, Thorndike, Skinner & Rescorla-Wagner
   - 6.2 Latent Learning, Cognitive Maps & The Tolman-Eichenbaum Machine (TEM)
   - 6.3 Taxonomies of Memory: Working Memory (Baddeley-Hitch, Cowan) & Long-Term Memory (Tulving)
   - 6.4 Attention, Sensory Gating & Bottlenecks: Broadbent, Treisman & Posner
   - 6.5 Predictive Processing, Active Inference & The Free Energy Principle (Helmholtz, Friston, Parr)
   - 6.6 Dual-Process Theory & Bounded Rationality: Simon, Kahneman & Tversky
7. [Module VI: Engineering Translation & Architecture Blueprints](#7-module-vi-engineering-translation--architecture-blueprints)
   - 7.1 Cross-Disciplinary Translation Matrix (Biology $\to$ Math $\to$ Computing)
   - 7.2 Overcoming Point-Neuron Limits: Compartmental SNNs vs. Phasor-VSA vs. Transformers
   - 7.3 Multi-Timescale Memory Hierarchy (T0 Phase, T1 Delta Residuals, T2 Semantic Consolidation)
   - 7.4 Algorithmic Blueprint: Building the Bio-Grounded Cognitive Unit & Column
   - 7.5 Zero-Cheat Benchmark Criteria for Autonomous Biological Learning
8. [Comprehensive Bibliography & Historical Milestones (Classic to 2026)](#8-comprehensive-bibliography--historical-milestones)

---

## 1. Operational Resilience & Interruption-Proof Workflow

> **CRITICAL OPERATING DIRECTIVE:** Due to recurrent power disruptions and environment resets, all computational plans, model checkpoints, experimental scorecards, and research documentation must operate under a **Zero-In-Memory-Only (ZIMO)** discipline.

### 1.1 Invariant Rules for Ongoing Work
1. **Immediate Disk Serialization:** Never accumulate multi-step results solely in transient interpreter memory. Every phase must append or commit directly to `.md`, `.json`, or checkpoint binaries on the physical workspace filesystem.
2. **Atomic & Idempotent Execution:** Every script and benchmark suite must be runnable from scratch with reproducible random seeds (`--seed=42`) and resume capabilities (checking if prior run outputs exist).
3. **Continuous Board & Artifact Mirroring:** Any changes to active tracks (Path A biological patching vs. Path B column scaling) must update `docs/BOARD.md` immediately upon card completion.
4. **Git Checkpoint Cadence:** Work must be committed to git branches at every logical card milestone to ensure uncorrupted state rollback.

---

## 2. Module I: Biophysical & Molecular Foundations of the Biological Neuron

```
               Extracellular Space (+60 mV Na+, +120 mV Ca2+, 0 mV Ref)
       ========================================================================
       [ Na+ Channel ]       [ K+ Channel ]       [ NMDA Receptor ]   [ Lipids ]
          |  ^                  |  ^                 |   ^   (Mg2+ block)   ||
          v  |                  v  |                 v   |        |         ||
       ========================================================================
               Intracellular Cytoplasm (-70 mV Resting, high K+, low Na+)
```

### 2.1 Membrane Biophysics, Resting Potentials & Passive Cable Theory
The biological neuron is bounded by a phospholipid bilayer dielectric ($C_m \approx 1.0 \, \mu\text{F/cm}^2$).

#### Electrochemical Equilibrium (Nernst Potential)
When an ion species $X$ of valence $z$ is distributed unevenly across a selectively permeable membrane, its thermodynamic equilibrium potential $E_X$ is defined by the Nernst equation:
$$E_X = \frac{RT}{zF} \ln \left( \frac{[X]_{\text{out}}}{[X]_{\text{in}}} \right)$$
At mammalian physiological temperature ($37^\circ\text{C} = 310.15\,\text{K}$):
- $E_{\text{K}^+} \approx -90\,\text{mV}$ (High inside: $\sim 140\,\text{mM}$ vs. out: $\sim 5\,\text{mM}$)
- $E_{\text{Na}^+} \approx +60\,\text{mV}$ (High outside: $\sim 145\,\text{mM}$ vs. in: $\sim 10\,\text{mM}$)
- $E_{\text{Ca}^{2+}} \approx +120\,\text{mV}$ (Sub-micromolar inside: $\sim 100\,\text{nM}$ vs. out: $\sim 2\,\text{mM}$)
- $E_{\text{Cl}^-} \approx -70\,\text{mV}$ (High outside: $\sim 110\,\text{mM}$ vs. in: $\sim 10\,\text{mM}$)

#### Steady-State Membrane Potential (Goldman-Hodgkin-Katz Voltage Equation)
When multiple ionic species permeate simultaneously:
$$V_m = \frac{RT}{F} \ln \left( \frac{P_{\text{K}}[\text{K}^+]_{\text{out}} + P_{\text{Na}}[\text{Na}^+]_{\text{out}} + P_{\text{Cl}}[\text{Cl}^-]_{\text{in}}}{P_{\text{K}}[\text{K}^+]_{\text{in}} + P_{\text{Na}}[\text{Na}^+]_{\text{in}} + P_{\text{Cl}}[\text{Cl}^-]_{\text{out}}} \right)$$
At rest, $P_{\text{K}} : P_{\text{Na}} : P_{\text{Cl}} \approx 1.0 : 0.04 : 0.45$, clamping somatic $V_{\text{rest}}$ between $-65\,\text{mV}$ and $-75\,\text{mV}$. The active $\text{Na}^+/\text{K}^+$-ATPase pump expels $3\,\text{Na}^+$ for every $2\,\text{K}^+$ imported, consuming $\sim 40-50\%$ of brain ATP.

#### Wilfrid Rall’s Linear Cable Theory
A cylindrical dendrite of radius $a$, intracellular axial resistivity $r_a$ ($\Omega\cdot\text{cm}$), and membrane resistance $r_m$ ($\Omega\cdot\text{cm}^2$) is governed by the 1D cable equation:
$$\lambda^2 \frac{\partial^2 V(x,t)}{\partial x^2} - \tau_m \frac{\partial V(x,t)}{\partial t} - (V(x,t) - V_{\text{rest}}) = 0$$
Where:
- **Electrotonic Space Constant:** $\lambda = \sqrt{\frac{a \cdot r_m}{2 r_a}}$ ($100\,\mu\text{m} - 1000\,\mu\text{m}$).
- **Membrane Time Constant:** $\tau_m = r_m c_m$ ($10\,\text{ms} - 50\,\text{ms}$).
- **Electrotonic Length:** $L = \ell / \lambda$. Under Rall's $3/2$ power rule ($d_{\text{parent}}^{3/2} = \sum d_{\text{daughter}}^{3/2}$), impedance match avoids reflection artifacts at bifurcations.

---

### 2.2 Ion Channels: Kinetics, Selectivity Filters & Channelopathies
1. **Voltage-Gated Sodium Channels ($\text{Na}_v1.1 - 1.9$):** Sub-millisecond activation via positively charged S4 transmembrane helices; inactivation via the cytosolic IFM (isoleucine-phenylalanine-methionine) motif plug.
2. **Voltage-Gated Potassium Channels ($\text{K}_v$):** Delayed rectifiers ($I_{\text{K}}$) mediate repolarization; A-type fast inactivating channels ($I_A$) control inter-spike intervals and spike latency.
3. **Voltage-Gated Calcium Channels ($\text{Ca}_v$):** High-voltage activated (L-type, P/Q-type, N-type) mediate neurotransmitter release and gene transcription; Low-voltage activated (T-type) mediate pacemaker oscillations and post-inhibitory rebound bursts.
4. **HCN Channels ($I_h$):** Reversal potential $\approx -30\,\text{mV}$. Activated by hyperpolarization; acts as a biological "voltage clamp" or restorative spring that dampens dendritic resonance and normalizes temporal summation along the apical trunk.
5. **Inward Rectifiers ($\text{K}_{\text{ir}}$):** Maintain hyperpolarized resting potentials without adding load during depolarization.

---

### 2.3 The Action Potential, Axon Initial Segment (AIS) & Active Backpropagation (bAP)

#### The Hodgkin-Huxley Formulation (1952)
$$I_m = C_m \frac{dV}{dt} + \bar{g}_{\text{Na}} m^3 h (V - E_{\text{Na}}) + \bar{g}_{\text{K}} n^4 (V - E_{\text{K}}) + g_L (V - E_L)$$
$$\frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x = \frac{x_\infty(V) - x}{\tau_x(V)} \quad \text{for } x \in \{m, h, n\}$$

#### The Axon Initial Segment (AIS)
Action potential initiation occurs at the **Axon Initial Segment (AIS)**—a specialized $20-60\,\mu\text{m}$ unmyelinated domain anchored by Ankyrin-G with over 100-fold higher density of $\text{Na}_v1.6$ channels compared to the soma, dropping the electrical threshold by $10-15\,\text{mV}$ to ensure directional, stereotypic spike firing.

#### Backpropagating Action Potentials (bAP)
Upon initiation at the AIS, spikes propagate:
1. **Orthodromic:** Down the axon to presynaptic terminals.
2. **Antidromic (bAP):** Retrogradely invade the soma and apical/basal dendrites via dendritic $\text{Na}_v$ channels, providing a sub-millisecond timestamp of somatic firing for Spike-Timing-Dependent Plasticity (STDP).

---

### 2.4 Dendritic Computation: Compartmentalization, Plateaus & Human dCaAPs (Gidon, Larkum, Poirazi)

```
            Apical Tuft (Layer 1) ---> Top-down Context / Feedback
                  |
             [ Ca2+ Spike Zone / Apical Nexus ] ---> Coincidence Detector
                  |
             Apical Trunk (HCN / Ih gradient)
                  |
             Soma & Basal Dendrites ---> Bottom-up Feedforward Signals
                  |
             Axon Initial Segment (AIS) ---> Output Spike (Action Potential)
```

#### Active Dendritic Phenomena (Chavlis & Poirazi, 2026; Shapira & Segev, 2025)
1. **Dendritic Sodium Spikes:** Rapid, local $Na^+$ spikes initiated in thin basal and oblique branches.
2. **NMDA Spikes / Voltage Plateaus:** Synchronous clustered glutamatergic inputs on a single branch expel the extracellular $\text{Mg}^{2+}$ ion from NMDA receptors ($+20\,\text{mV}$ to $+30\,\text{mV}$ local threshold). This triggers a sustained plateau depolarization lasting $50-200\,\text{ms}$, injecting a steady plateau current ($I_{\text{inj}} \approx 1-3\,\text{nA}$) into the parent dendrite.
3. **Calcium Nexus Spikes:** In Layer 5 pyramidal neurons, the apical nexus expresses high-voltage-activated $\text{Ca}_v$ channels. Synchronous tuft input evokes long-duration ($>30\,\text{ms}$) dendritic calcium action potentials.

#### Two-Layer Hierarchical Network Model (Poirazi, Brannon & Mel, 2003)
A single pyramidal neuron acts mathematically as a **two-layer artificial neural network**:
$$V_{\text{soma}}(t) = g\left( \sum_{j=1}^B f_j\left( \sum_{i \in \mathcal{S}_j} w_{ij} x_i(t) \right) \right)$$
Where $f_j(\cdot)$ represents local non-linear branch summation (NMDA plateau thresholding), and $g(\cdot)$ is somatic spike thresholding.

#### Human Dendritic Action Potentials (dCaAPs) & Single-Neuron XOR (Gidon et al., 2020)
Recordings from human Layer 2/3 cortical pyramidal cells reveal:
- Human L2/3 dendrites exhibit novel **dendritic Calcium Action Potentials (dCaAPs)** whose amplitude is **graded and inversely monotonic** with input intensity (maximal near threshold, suppressed under higher drive).
- **Single-Neuron Non-Separability:** This property allows an individual human neuron to compute the **exclusive-OR (XOR)** logical operation without intermediate hidden neurons.

---

### 2.5 Synaptic Physiology: Quantal Neurotransmission & Vesicle Dynamics
- **Quantal Hypothesis (Katz, 1969):** Mean response $v = n \cdot p \cdot q$ ($n$ release sites, $p$ exocytosis probability, $q$ quantal size $\approx 1000-5000$ glutamate molecules).
- **SNARE Fusion:** Depolarization triggers $\text{Ca}^{2+}$ nanodomains ($>10\,\mu\text{M}$); calcium binds **Synaptotagmin-1**, releasing the clamp and zippering Syntaxin-1, SNAP-25, and Synaptobrevin/VAMP into a 4-helix bundle that executes membrane fusion in $<200\,\mu\text{s}$.
- **Vesicle Pools:** Readily Releasable Pool (RRP, $1-2\%$), Recycling Pool ($10-20\%$), Reserve Pool ($80-90\%$).
- **Receptor Subtypes:**
  - **AMPA (GluA1-4):** Fast cation flux ($Na^+, K^+$); $\tau_{\text{decay}} \approx 1-5\,\text{ms}$.
  - **NMDA (GluN1, GluN2A-D):** Voltage-dependent $\text{Mg}^{2+}$ block (Woodhull equation: $B(V) = [1 + \frac{[\text{Mg}^{2+}]_o}{3.57} e^{-0.062 V}]^{-1}$); high $Ca^{2+}$ permeability; $\tau_{\text{decay}} \approx 50-250\,\text{ms}$.
  - **$\text{GABA}_A$:** Fast ionotropic $Cl^-$ channel ($\tau \approx 5-15\,\text{ms}$).
  - **$\text{GABA}_B$:** Metabotropic GPCR activating GIRK potassium channels; slow shunting inhibition ($\tau \approx 100-500\,\text{ms}$).
- **Tsodyks-Markram Dynamic Synapses:**
  $$\frac{dR(t)}{dt} = \frac{1 - R(t)}{\tau_{\text{rec}}} - u(t)^+ R(t) \cdot \delta(t - t_k), \quad \frac{du(t)}{dt} = \frac{U - u(t)}{\tau_{\text{fac}}} + U(1 - u(t)) \cdot \delta(t - t_k)$$
  $$I_{\text{syn}}(t) = A \cdot R(t) \cdot u(t) \cdot \delta(t - t_k)$$
  Mediates Short-Term Depression (STD: high $U$, novelty detector) and Short-Term Facilitation (STF: low $U$, temporal integrator).

---

### 2.6 The Tripartite Synapse & Glial Biology
The human brain contains $\sim 86$ billion neurons and $\sim 85$ billion glia.
- **Astrocytic Coverage:** A single human protoplasmic astrocyte enwraps up to $2,000,000$ synapses.
- **Glutamate Clearance:** EAAT1/GLT-1 and EAAT2/GLAST transporters clear glutamate within milliseconds, preventing excitotoxicity.
- **Astrocyte-Neuron Lactate Shuttle (ANLS):** Astrocytes convert glucose into lactate and export it via MCTs to neurons for mitochondrial energy.
- **Gliotransmission & Calcium Microdomains:** Non-spiking astrocytic $Ca^{2+}$ microdomain waves trigger release of D-serine, glutamate, and ATP.
- **Homeostatic Purinergic Scaling:** Extracellular ATP degrades to **adenosine**, which binds presynaptic $A_1$ receptors to scale down release probability homeostatically across circuits.
- **Microglia & Oligodendrocytes:** Microglia prune complement-tagged ($C1q$) weak synapses; oligodendrocytes synthesize myelin sheaths, enabling saltatory conduction up to $>100\,\text{m/s}$.

---

## 3. Module II: Synaptic Plasticity, Metaplasticity & Neuromodulation

### 3.1 Hebbian Learning & Spike-Timing-Dependent Plasticity (STDP)

#### Classic Bipolar STDP (Bi & Poo, 1998)
$$\Delta w = \begin{cases} 
A_+ \exp\left(-\frac{\Delta t}{\tau_+}\right) & \text{if } \Delta t > 0 \quad (\text{LTP}) \\
-A_- \exp\left(\frac{\Delta t}{\tau_-}\right) & \text{if } \Delta t < 0 \quad (\text{LTD})
\end{cases}$$
Where $\tau_+ \approx \tau_- \approx 15-25\,\text{ms}$, and $A_- \tau_- > A_+ \tau_+$ guarantees stability under uncorrelated Poisson spike noise.
- **LTP Pathway:** High calcium ($>5\,\mu\text{M}$) activates **CaMKII** autophosphorylation at Thr286, driving AMPA receptor exocytosis.
- **LTD Pathway:** Low, sustained calcium ($0.5-1.0\,\mu\text{M}$) activates phosphatases (**Calcineurin / PP2B** and **PP1**), causing AMPA receptor endocytosis.

---

### 3.2 Burst-Dependent Plasticity & Somato-Dendritic Decoupling (Naud et al., 2021)
- Isolated somatic spikes evoke negligible plasticity.
- Coincidence of an apical dendritic calcium spike and somatic firing produces a high-frequency **burst**.
- Basal dendrites drive somatic action potentials (feedforward sensory data), while the apical tuft encodes top-down predictive feedback. The burst rate selectively carries the error/feedback signal, enabling local multi-layer credit assignment without backpropagation.

---

### 3.3 Three-Factor Learning Rules & Neuromodulatory Systems
To solve the distal reward credit assignment problem:
$$\Delta w_{ij}(t) = \eta \cdot e_{ij}(t) \cdot M(t)$$
$$\tau_e \frac{de_{ij}(t)}{dt} = -e_{ij}(t) + f(\text{pre}_i(t), \text{post}_j(t))$$
Where $e_{ij}(t)$ is an eligibility trace and $M(t)$ is a global neuromodulator:
1. **Dopamine (DA):** Encodes Reward Prediction Error (RPE; Schultz, Dayan & Montague, 1997). Gates R-STDP; stabilizes LTP at D1 receptors and LTD at D2 receptors.
2. **Acetylcholine (ACh):** Suppresses recurrent intracortical transmission while enhancing feedforward afferents (Hasselmo); sets the signal-to-noise ratio and gates receptive field plasticity.
3. **Noradrenaline / Norepinephrine (NE):** Signals unexpected surprise and volatility (Aston-Jones & Cohen); adjusts global neural gain and resets attractor states.
4. **Serotonin (5-HT):** Modulates temporal discounting and behavioral inhibition for delayed payoffs.

---

### 3.4 Homeostatic Plasticity, BCM Sliding Thresholds & Synaptic Scaling

#### The BCM Theory (Bienenstock, Cooper & Munro, 1982)
$$\frac{dw_i}{dt} = \eta \cdot x_i \cdot y \cdot (y - \theta_M), \quad \tau_\theta \frac{d\theta_M}{dt} = -\theta_M + \left(\frac{y}{y_0}\right)^p y_0 \quad (p > 1)$$
The modification threshold $\theta_M$ slides dynamically as a super-linear function of historical firing, preventing epileptic runaway and silence.

#### Multiplicative Synaptic Scaling (Gina Turrigiano, 1998)
$$w_i(t + \Delta t) = w_i(t) \cdot \left( 1 + \gamma \left( R_{\text{target}} - \bar{R} \right) \right)$$
Adjusts all excitatory inputs multiplicatively to maintain stable target firing while preserving relative synaptic weight ratios.

---

### 3.5 Structural Plasticity: Spine Remodeling, Pruning & Silent Synapses
- **Spine Dynamics:** Motile, plastic "learning spines" (thin) mature into stable, anchored "memory spines" (mushroom).
- **Silent Synapses:** Spines expressing NMDA receptors without AMPA receptors. LTP rapidly recruits AMPA receptors, "unsilencing" the synapse within minutes.
- **Complement-Mediated Pruning:** Microglia engulf inactive, complement-tagged ($C1q$) synapses to maintain network sparsity.

---

## 4. Module III: Neuroanatomy, Laminar Microcircuits & Cortical Columns

```
                     NEOCORTICAL 6-LAYER LAMINAR ARCHITECTURE
  ================================================================================
  L1 (Molecular)      | Apical tufts, horizontal axons, matrix thalamus, VIP interneurons
  --------------------+-----------------------------------------------------------
  L2/3 (Supragranular)| Pyramidal cells, dense lateral corticocortical recurrents
                      | Computes: Feature conjunctions, feedforward prediction errors
  --------------------+-----------------------------------------------------------
  L4 (Granular)       | Spiny stellate cells; primary target of core thalamic relay
                      | Computes: Primary sensory feedforward feature extraction
  --------------------+-----------------------------------------------------------
  L5 (Infragranular)  | Large burst-firing pyramidal neurons; motor/striatal output
                      | Computes: Predictions, motor commands, somatic-apical coincidence
  --------------------+-----------------------------------------------------------
  L6 (Polymorphic)    | Corticothalamic feedback neurons; deep reciprocal loops
                      | Computes: Modulatory gain control of thalamus, sensory routing
  ================================================================================
```

### 4.1 Neocortical Laminar Cytoarchitecture (Layers 1 to 6)
- **L1:** Devoid of somata; apical tufts, horizontal feedback axons, and inhibitory interneurons.
- **L2/3:** Pyramidal cells with dense lateral recurrents; feedforward output to higher cortical areas.
- **L4:** Spiny stellates receiving specific thalamocortical inputs (core sensory gateway).
- **L5:** Thick-tufted pyramidal neurons projecting to subcortical motor/action targets (striatum, colliculus, spinal cord).
- **L6:** Reciprocal corticothalamic feedback modulating thalamic gating.

---

### 4.2 Canonical Microcircuits: Mountcastle, Szentágothai, Douglas & Martin
- **Minicolumn:** Vertical core of $\sim 80-120$ neurons ($20-50\,\mu\text{m}$ diameter) sharing receptive field properties.
- **Macrocolumn:** Cluster of minicolumns bound by lateral inhibition ($300-500\,\mu\text{m}$ diameter).
- **Canonical Flow (Douglas & Martin, 1989):**
  $$\text{Thalamus} \longrightarrow \text{Layer 4} \longrightarrow \text{Layer 2/3} \longrightarrow \text{Layer 5} \longrightarrow \text{Layer 6} \longrightarrow \text{Thalamus}$$
  Over $80\%$ of excitatory synapses originate within local recurrent circuits.

---

### 4.3 Predictive Coding Microcircuits: Bastos, Friston & Larkum
- **Feedforward Streams Carry Errors ($e$):** Generated in L2/3; ascend to L4 of higher areas; synchronized in the **gamma band ($30-80\,\text{Hz}$)**.
- **Feedback Streams Carry Predictions ($\mu$):** Generated in L5/6; descend to L1 and deep layers; synchronized in **alpha/beta bands ($8-30\,\text{Hz}$)**.
- **Apical Coincidence Detection (Larkum, 2013):** L5 apical tufts receive top-down context, while basal dendrites receive sensory evidence. Coincidence triggers a dendritic calcium plateau and high-frequency somatic burst firing.

---

### 4.4 The Cortical Column & Thousand Brains Framework (Hawkins / Numenta)
- **Reference Frames Everywhere:** Every cortical column learns complete sensorimotor models of objects by routing inputs into metric reference frames (analogous to grid cells).
- **Sensorimotor Loop:** Perception integrates sensory updates with motor displacement vectors ($\Delta x$).
- **Long-Range Voting:** Independent columns vote via sparse lateral L2/3 projections to achieve perceptual consensus.

---

### 4.5 Subcortical Loops: Basal Ganglia Action Selection & Thalamocortical Gating
- **Direct Pathway ("Go"):** $\text{Cortex} \to \text{Striatum (D1)} \xrightarrow{\text{GABA}} \text{GPi/SNr} \xrightarrow{\text{disinhibition}} \text{Thalamus} \to \text{Cortex}$. Releases action.
- **Indirect Pathway ("No-Go"):** $\text{Cortex} \to \text{Striatum (D2)} \xrightarrow{\text{GABA}} \text{GPe} \xrightarrow{\text{GABA}} \text{STN} \xrightarrow{\text{Glu}} \text{GPi/SNr} \to \text{Thalamus}$. Clamps competing motor plans.
- **Dopamine Modulation:** Dopamine excites Direct (D1) and inhibits Indirect (D2) pathways, biasing the network toward action initiation.
- **Higher-Order Thalamus (Pulvinar):** Drives trans-cortical communication between Layer 5 and higher-order cortical Layer 1.

---

### 4.6 The Cerebellar Microcircuit: Marr-Albus-Ito Models
- **Architecture:** Mossy fibers $\to$ Granule cells / Parallel fibers $\to$ Purkinje cells $\leftarrow$ Climbing fibers.
- **Plasticity:** Climbing fibers deliver complex spike error signals. Coincident parallel fiber firing induces **LTD** at parallel fiber-Purkinje synapses.
- **Computation:** Learns predictive **forward internal models** that cancel out self-generated sensory feedback (*efference copy*) and execute feedforward motor trajectories.

---

## 5. Module IV: Systems Neuroscience, Dynamical Coding & Brain Rhythms

### 5.1 Electrophysiological Rhythms & Spectral Bands
- **Delta ($0.5-4\,\text{Hz}$):** Deep non-REM sleep; slow-wave memory consolidation.
- **Theta ($4-8\,\text{Hz}$):** Hippocampal navigation; episodic sequencing; slot buffering.
- **Alpha ($8-12\,\text{Hz}$):** Gating by inhibition of task-irrelevant regions.
- **Beta ($13-30\,\text{Hz}$):** Top-down status quo maintenance; motor stabilization.
- **Gamma ($30-80+\,\text{Hz}$):** Local PING interneuron rhythm; feature binding; bottom-up error signaling.

---

### 5.2 Synchrony & Binding-by-Synchrony (Singer & Fries)
- **Temporal Binding Hypothesis (Singer, 1999):** Distributed assemblies responding to the same perceptual object synchronize action potentials within a millisecond window in the gamma band ($40\,\text{Hz}$).
- **Communication-Through-Coherence (Fries, 2005 / 2015):** Transmission between regions succeeds only when the sender's spike bursts arrive during the receiver's peak rhythmic excitability phase.

---

### 5.3 Dynamical Systems: Kuramoto Coupled Oscillators & CANNs

#### The Kuramoto Model (1975)
$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i), \quad r(t) e^{i \psi(t)} = \frac{1}{N} \sum_{j=1}^N e^{i \theta_j(t)}$$
When coupling $K$ exceeds $K_c$, the network undergoes a second-order phase transition into collective phase-locking ($r \to 1$).

#### Continuous Attractor Neural Networks (CANNs)
$$\tau \frac{du(x,t)}{dt} = -u(x,t) + \int W(x - x') \sigma(u(x',t)) dx' + I_{\text{ext}}(x,t)$$
Symmetric Mexican-hat connectivity forms a stable localized activity bump that tracks continuous physical and abstract coordinates.

---

### 5.4 Spatial Navigation & Metric Cognitive Maps
- **Place Cells (CA1/CA3):** Topological location encoding.
- **Grid Cells (MEC L2/3; Moser & Moser, 2005):** Hexagonal metric coordinate grid across discrete modular scales ($\lambda_{i+1} / \lambda_i \approx 1.4$). Integrates velocity inputs to maintain a continuous coordinate manifold.
- **Head-Direction Cells & Border Cells:** Directional compass and boundary distance metrics.

---

### 5.5 Phase Precession, Theta-Gamma PAC & Working Memory Slots
- **Phase Precession (O'Keefe & Recce, 1993):** Spikes precess systematically from late to early phases of the hippocampal theta cycle as an animal traverses a place field, doubling spatial readout precision.
- **Theta-Gamma Slot Coding (Lisman & Idiart, 1995):** A single theta cycle ($\sim 150\,\text{ms}$) accommodates $5-8$ gamma sub-cycles ($\sim 25\,\text{ms}$), each hosting one active item assembly—the biophysical basis of Miller's $7 \pm 2$ working memory capacity.

---

### 5.6 Memory Consolidation & Complementary Learning Systems (CLS)
- **Two-Stage Memory Architecture (McClelland et al., 1995):** Fast hippocampal buffer (sparse, high plasticity, one-shot episodic capture) $\leftrightarrow$ Slow neocortical learner (distributed, small learning rate, semantic abstraction).
- **Sharp-Wave Ripples (SWRs; 150-250 Hz):** Compressed forward and reverse trajectory replay ($10-20\times$ speedup) during non-REM sleep and quiet rest. Recent evidence confirms that **large, high-amplitude SWR clusters** are selectively upregulated to drive consolidation into prefrontal cortex without catastrophic interference.

---

### 5.7 Continuous Temporal Memory: Hippocampal Time Cells & State-Space Projection (LMU)
- **Time Cells (Fried et al., Nature; Eichenbaum):** CA1 neurons fire at stereotypic time intervals across delay periods, bridging temporal gaps to associate disjoint events.
- **State-Space History Projection (Legendre Memory Unit; Voelker & Eliasmith, 2018):**
  $$\dot{h}(t) = A h(t) + B x(t)$$
  $$A_{ij} = \frac{2i + 1}{\theta} \begin{cases} -1 & i < j \\ (-1)^{i-j+1} & i \ge j \end{cases}, \quad B_i = \frac{2i+1}{\theta} (-1)^i$$
  Compresses an analog continuous history window of length $\theta$ onto an orthogonal Legendre polynomial basis of degree $d$, replacing $O(T)$ tapped delay lines with $O(d)$ state variables.

---

## 6. Module V: Psychology, Cognitive Science & Foundational Principles of Mind

### 6.1 Classical & Operant Conditioning
- **Pavlovian Conditioning (1927):** Pairing neutral CS with potent US drives conditioned response (CR).
- **The Rescorla-Wagner Model (1972):**
  $$\Delta V_i = \alpha_i \beta \left( \lambda - \sum_{j} V_j \right)$$
  Learning is driven by **associative prediction error**. Explains Kamin blocking: prior predictive validity blocks associative updates to redundant redundant cues.
- **Operant Conditioning (Skinner):** Variable-ratio schedules drive the highest response rates and resistance to extinction.

---

### 6.2 Latent Learning, Cognitive Maps & The Tolman-Eichenbaum Machine (TEM)
- **Latent Learning (Tolman, 1948):** Animals acquire latent spatial models without reward, deploying them instantly when motivation appears.
- **The Tolman-Eichenbaum Machine (Whittington & Behrens, 2020, Cell):**
  $$p(x_t \mid x_{<t}) = \sum_{g} p(x_t \mid g_t) p(g_t \mid g_{t-1}, a_t)$$
  Separates structural relational knowledge $g_t$ (entorhinal grid cell basis / graph transitions) from sensory particulars $x_t$ (hippocampal representations), enabling zero-shot structural generalization across non-spatial domains (social hierarchies, concept trees, family relationships).

---

### 6.3 Taxonomies of Memory: Working & Long-Term Memory
- **Baddeley & Hitch Model (1974 / 2000):** Central Executive, Phonological Loop, Visuospatial Sketchpad, and Episodic Buffer.
- **Cowan’s Embedded-Process Model (2005):** Working memory as the focus of attention ($4 \pm 1$ items) acting upon the activated subset of long-term memory.
- **Tulving’s Taxonomy:** Declarative (Episodic autonoetic, Semantic noetic) vs. Non-Declarative (Procedural anoetic, Priming, Conditioning).

---

### 6.4 Attention & Sensory Gating
- **Broadbent Early Filter:** Strict bottleneck admitting only attended channels.
- **Treisman Feature Integration (1980):** Parallel pre-attentive feature detection; serial attentional spotlight required to bind features (color + shape).
- **Posner Networks:** Alerting (NE), Orienting (ACh), Executive Control (frontoparietal dopamine).

---

### 6.5 Predictive Processing, Active Inference & The Free Energy Principle (FEP)

#### Variational Free Energy Minimization (Friston, 2010)
$$\mathcal{F} = \mathbb{E}_{q(\eta)} [\ln q(\eta) - \ln p(\eta, y)] = D_{\text{KL}}[q(\eta) \mid\mid p(\eta \mid y)] - \ln p(y)$$
Organisms maintain homeostasis by minimizing sensory surprise via two complementary mechanisms:
1. **Perception:** Update internal beliefs $\mu$ to fit sensations $y$.
2. **Action (Active Inference):** Act on the world to make observations conform to internal expectations.

#### Expected Free Energy for Planning (Parr & Friston, 2022)
$$\mathbf{G}(\pi) = \underbrace{\mathbb{E}[\ln q(\eta \mid y, \pi) - \ln q(\eta \mid \pi)]}_{\text{Epistemic Value (Information Gain / Exploration)}} + \underbrace{\mathbb{E}[\ln p(y) - \ln q(y \mid \pi)]}_{\text{Pragmatic Value (Utility / Goal Exploitation)}}$$
Unifies exploratory drive and reward pursuit into a single variational planning objective.

---

### 6.6 Dual-Process Theory & Bounded Rationality
- **Bounded Rationality (Simon, 1957):** Satisficing under finite computational constraints rather than global utility maximization.
- **Dual-Process Modes (Kahneman & Tversky, 2011):**
  - **System 1:** Fast, intuitive, parallel, heuristic, associative.
  - **System 2:** Slow, deliberative, serial, rule-governed, computationally expensive.

---

## 7. Module VI: Engineering Translation & Architecture Blueprints

### 7.1 Cross-Disciplinary Translation Matrix

```
+-----------------------------------+------------------------------------+---------------------------------------+
| Biological & Cognitive Mechanism  | Mathematical Formulation           | Engineered Computational Primitive   |
+-----------------------------------+------------------------------------+---------------------------------------+
| Somatic Integration & Spiking     | Leaky Integrate-and-Fire / HH      | LIF / Resonate-and-Fire / Phasor SNN  |
| Active Dendritic Plateaus (NMDA)  | 2-layer Branch Non-linearity       | Compartmental Sub-networks / Gating   |
| Human dCaAP Graded Amplitude      | Inverse Monotonic Threshold        | Single-Neuron Non-linear Logic (XOR)  |
| Astrocytic Gain & Metaplasticity  | Homeostatic Feedback Loop          | Gated Synaptic Scaling / Dynamic Bias |
| Short-Term Plasticity (STP)       | Tsodyks-Markram (R, u variables)   | Dynamic Attention / Recurrent Fast W  |
| Dopaminergic Three-Factor Learning| Δw = η · e_ij · RPE                | Reinforcement-STDP / Policy Gradient  |
| Associative Surprise Metric       | Rescorla-Wagner / Free Energy F    | Precision-Weighted Prediction Error   |
| Metric Spatial Navigation         | Continuous Attractor Manifold      | Toroidal CANN / Vector Symbolic Frame |
| Theta-Gamma Slot Coding           | Phase-Amplitude Coupling           | Oscillatory Working Memory Buffers    |
| Two-Stage Systems Consolidation   | Fast Sparse Buffer + Slow Abstr.   | Tiered Memory (T0 -> T1 -> T2 Replay) |
| Multi-Modal Binding & Gesture     | Binding-by-Synchrony               | Complex Phasor VSA Bind / Unbind      |
| Cortical Consensus                | Layer 2/3 Lateral Voting           | Cortical Messaging Protocol / Voting  |
+-----------------------------------+------------------------------------+---------------------------------------+
```

---

### 7.2 Overcoming Point-Neuron Limits: Why Single LIF Cells Fail
Point-neuron models (Perceptrons, standard LIF) fail biological reality in three ways:
1. **Computational Bottleneck:** They cannot solve linearly non-separable tasks (XOR, directional selectivity) without adding artificial hidden layers.
2. **Catastrophic Forgetting:** Overwriting single weight matrices destroys prior knowledge during continuous streaming.
3. **Temporal Amnesia:** Lacking intermediate memory timescales, they cannot bridge delays without external buffer lines.

---

### 7.3 Multi-Timescale Memory Hierarchy: The RPN / Cognitive Unit Blueprint

```
                          THREE-TIMESCALE MEMORY FLOW
  ================================================================================
  [ Perceived Signal ] 
          |
          v
  +------------------------------------------------------------------------------+
  | T0: WORKING MEMORY (Sub-second)                                              |
  | Substrate: Transient Oscillatory Phase Pattern (Theta slot, Gamma item)       |
  | Dynamics: Kuramoto Resonance / Limit-Cycle Settling (k micro-steps)          |
  +------------------------------------------------------------------------------+
          |
          v (Gated strictly by SURPRISE: 1 - sim(prediction, target))
  +------------------------------------------------------------------------------+
  | T1: EPISODIC FAST-WEIGHTS (Seconds to Minutes)                               |
  | Substrate: Gated Delta-Rule Matrix: W += η · outer(target - W@ctx, conj(ctx))|
  | Property: Stores residuals only -> NEVER saturates like additive bundling   |
  | Dynamics: Exponential / Adaptive Ebbinghaus Decay                            |
  +------------------------------------------------------------------------------+
          |
          v (Offline Systems Consolidation / Replay)
  +------------------------------------------------------------------------------+
  | T2: CONSOLIDATED SEMANTIC STORE (Long-term / Indefinite)                     |
  | Substrate: High-dimensional Phasor Basis / Stable Attractors                 |
  | Property: Robust to catastrophic forgetting (BWT >= 0.0)                     |
  +------------------------------------------------------------------------------+
```

---

### 7.4 Algorithmic Blueprint: Building the Bio-Grounded Cognitive Unit & Column

Here is the exact mathematical algorithm for one complete forward, learning, and consolidation cycle of an engineered **Resonant Predictive Column (RPC)**:

```python
class ResonantPredictiveUnit:
    """
    Atomic cognitive unit unifying:
    - Multi-band oscillatory dendrites (T0 Working Memory)
    - Gated Delta-Rule fast weights (T1 Episodic Memory)
    - Consolidated semantic attractors (T2 Semantic Memory)
    - Surprise-gated prediction loop (Active Inference)
    - Rectified Born gate (Anti-hallucination confidence filter)
    """
    def __init__(self, D: int, K_bands: int = 3, lr_delta: float = 0.2):
        self.D = D                       # Hypervector dimension (e.g. 512, complex64)
        self.K = K_bands                 # Number of dendritic oscillatory bands (Gamma/Beta/Alpha)
        self.lr = lr_delta               # Learning rate for residual updates
        self.W_t1 = np.zeros((D, D), dtype=np.complex64)  # T1 episodic fast-weights
        self.W_t2 = np.zeros((D, D), dtype=np.complex64)  # T2 consolidated semantic store

    def predict(self, ctx_hv: np.ndarray, k_settle: int = 4) -> np.ndarray:
        """Settle oscillatory state for k micro-steps and predict next state."""
        h = ctx_hv.copy()
        # Kuramoto-style / relaxation settling loop (k steps)
        for _ in range(k_settle):
            h_pred = self.W_t1 @ h + self.W_t2 @ h
            h = normalise(h + 0.1 * h_pred)
        return h_pred

    def learn(self, ctx_hv: np.ndarray, target_hv: np.ndarray, surprise_thresh: float = 0.2):
        """Surprise-gated Gated Delta-Rule write (stores residuals only)."""
        # Step 1: Predict expected target from context
        predicted = self.predict(ctx_hv, k_settle=1)
        sim = float(np.abs(np.vdot(normalise(predicted), normalise(target_hv))))
        surprise = 1.0 - sim

        # Step 2: Surprise gate (Titans / Predictive Coding principle)
        if surprise > surprise_thresh:
            # Step 3: Compute residual error (Gated Delta Net / Delta Memory rule)
            residual = target_hv - (self.W_t1 @ ctx_hv)
            # Step 4: Write residual into T1 fast-weights
            self.W_t1 += self.lr * np.outer(residual, np.conj(ctx_hv))

    def consolidate(self, alpha_replay: float = 0.05):
        """Systems consolidation (Sharp-Wave Ripple analog): transfer T1 -> T2."""
        self.W_t2 += alpha_replay * self.W_t1
        self.W_t1 *= (1.0 - alpha_replay)  # Ebbinghaus decay of episodic traces
```

---

### 7.5 Zero-Cheat Benchmark Criteria for Autonomous Biological Learning

1. **Random Synapse Initialization & Structural Symmetry Breaking:** Synapses must initialize randomly. Tasks like XOR must converge via local plasticity (R-STDP / Delta-rule) breaking symmetry autonomously without hardcoded routing.
2. **Dynamic Delay Discovery:** The neuron must discover temporal delays autonomously via state-space projection (LMU) or dynamical resonance, not hand-coded readout timing offsets.
3. **Real Control Baselines:** Every benchmark must compare against architecturally matched external baselines (2-layer SNNs, Online MLPs, LSTMs, monolithic units).
4. **Statistical Rigor:** All experiments must execute $N \ge 20-100$ random seeds, reporting median and interquartile ranges (IQR) for non-Gaussian distributions with explicit OOD validation.

---

## 8. Comprehensive Bibliography & Historical Milestones

### Foundational Neurobiology & Cellular Biophysics
- **Hodgkin, A. L., & Huxley, A. F. (1952).** A quantitative description of membrane current and its application to conduction and excitation in nerve. *The Journal of Physiology*, 117(4), 500-544.
- **Rall, W. (1959).** Branching dendritic trees and motoneuron membrane resistivity. *Experimental Neurology*, 1(5), 491-527.
- **Katz, B. (1969).** *The Release of Neural Transmitter Substances*. Charles C Thomas.
- **Poirazi, P., Brannon, T., & Mel, B. W. (2003).** Pyramidal neuron as two-layer neural network. *Neuron*, 37(6), 989-999.
- **Gidon, A., et al. (2020).** Dendritic action potentials and computation in human layer 2/3 cortical pyramidal neurons. *Science*, 367(6473), 83-87.
- **Shapira, S., Segev, I., London, M., et al. (2025).** Biophysical and computational insights from modeling human cortical pyramidal neurons. *Frontiers in Neuroscience*, 19, 1421092.
- **Chavlis, S., & Poirazi, P. (2026).** A dendro-centric view of cognition in the behaving brain. *Annual Review of Neuroscience*, 49.
- **Araque, A., et al. (1999).** Tripartite synapses: glia, the unacknowledged partner. *Trends in Neurosciences*, 22(5), 208-215.
- **Tsodyks, M. V., & Markram, H. (1997).** The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *PNAS*, 94(2), 719-723.

### Synaptic Plasticity, Metaplasticity & Neuromodulation
- **Hebb, D. O. (1949).** *The Organization of Behavior: A Neuropsychological Theory*. John Wiley & Sons.
- **Bi, G. Q., & Poo, M. M. (1998).** Synaptic modifications in cultured hippocampal neurons: dependence on spike timing and biophysical properties. *The Journal of Neuroscience*, 18(24), 10464-10472.
- **Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982).** Theory for the development of neuron selectivity. *The Journal of Neuroscience*, 2(1), 32-48.
- **Turrigiano, G. G., et al. (1998).** Activity-dependent scaling of quantal AMPA receptor amplitude in cultured visual cortex. *Nature*, 391(6670), 892-896.
- **Schultz, W., Dayan, P., & Montague, P. R. (1997).** A neural substrate of prediction and reward. *Science*, 275(5306), 1593-1599.
- **Naud, R., Payeur, A., Richards, B. A., et al. (2021).** Burst-dependent synaptic plasticity coordinates learning in hierarchical circuits. *Nature Neuroscience*, 24(7), 1010-1019.
- **Hasselmo, M. E. (2006).** The role of acetylcholine in learning and memory. *Current Opinion in Neurobiology*, 16(6), 710-715.
- **Aston-Jones, G., & Cohen, J. D. (2005).** An integrative theory of locus coeruleus-norepinephrine function. *Annual Review of Neuroscience*, 28, 403-450.

### Laminar Cortical Circuits, Columns & Predictive Processing
- **Mountcastle, V. B. (1957).** Modality and topographic properties of single neurons of cat's somatic sensory cortex. *Journal of Neurophysiology*, 20(4), 408-434.
- **Douglas, R. J., & Martin, K. A. (1989).** A minimal cortical circuit of language and perception. *Neural Computation*, 1(4), 480-488.
- **Bastos, A. M., Usrey, W. M., Adams, R. A., Mangun, G. R., Fries, P., & Friston, K. J. (2012).** Canonical microcircuits for predictive coding. *Neuron*, 76(4), 695-711.
- **Larkum, M. (2013).** A cellular mechanism for cortical associations: an organizing principle for the cerebral cortex. *Trends in Neurosciences*, 36(3), 141-151.
- **Hawkins, J., & Ahmad, S. (2016).** Why neurons have thousands of synapses, a theory of sequence memory in neocortex. *Frontiers in Neural Circuits*, 10, 23.
- **Hawkins, J. (2021).** *A Thousand Brains: A New Theory of Intelligence*. Basic Books.

### Systems Neuroscience, Brain Rhythms & Cognitive Maps
- **Singer, W. (1999).** Neuronal synchrony: a versatile code for the definition of relations? *Neuron*, 24(1), 49-65.
- **Fries, P. (2005 / 2015).** Rhythms for cognition: communication through coherence. *Neuron*, 88(1), 220-235.
- **Kuramoto, Y. (1975).** Self-entrainment of a population of coupled non-linear oscillators. *Int. Symp. on Math. Problems in Theoretical Physics*, 420-422.
- **O'Keefe, J., & Nadel, L. (1978).** *The Hippocampus as a Cognitive Map*. Oxford University Press.
- **Moser, E. I., Kropff, E., & Moser, M. B. (2008).** Place cells, grid cells, and the brain's spatial representation system. *Annual Review of Neuroscience*, 31, 69-89.
- **O'Keefe, J., & Recce, M. L. (1993).** Phase relationship between hippocampal place units and the EEG theta rhythm. *Hippocampus*, 3(3), 317-330.
- **Lisman, J. E., & Idiart, M. A. (1995).** Storage of 7 +/- 2 short-term memories in oscillatory subcycles. *Science*, 267(5203), 1512-1515.
- **McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995).** Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419-457.
- **Voelker, A. R., & Eliasmith, C. (2018).** Improving spiking dynamical networks: A Legendre memory unit. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.

### Cognitive Psychology & Cognitive Science
- **Pavlov, I. P. (1927).** *Conditioned Reflexes*. Oxford University Press.
- **Rescorla, R. A., & Wagner, A. R. (1972).** A theory of Pavlovian conditioning. *Classical Conditioning II*, 64-99.
- **Tolman, E. C. (1948).** Cognitive maps in rats and men. *Psychological Review*, 55(4), 189-208.
- **Whittington, J. C. R., Muller, T. H., Mark, S., Chen, G., Barry, C., Burgess, N., & Behrens, T. E. J. (2020).** The Tolman-Eichenbaum Machine: Unifying space and relational memory through generalization in the hippocampal formation. *Cell*, 183(5), 1249-1263.
- **Baddeley, A. D., & Hitch, G. (1974).** Working memory. *Psychology of Learning and Motivation*, 8, 47-89.
- **Tulving, E. (1972).** Episodic and semantic memory. *Organization of Memory*, 381-403.
- **Treisman, A. M., & Gelade, G. (1980).** A feature-integration theory of attention. *Cognitive Psychology*, 12(1), 97-136.
- **Friston, K. (2010).** The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- **Parr, T., Pezzulo, G., & Friston, K. J. (2022).** *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
- **Simon, H. A. (1957).** *Models of Man, Social and Rational*. John Wiley & Sons.
- **Kahneman, D. (2011).** *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
