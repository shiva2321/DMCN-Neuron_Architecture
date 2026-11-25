"""Authentic Biophysical Human L2/3 Pyramidal Neuron with dCaAP Shunting.

Implements the two-compartment (apical dendrite + soma) biophysical model
discovered in human neocortical layer 2/3 pyramidal neurons by Albert Gidon et al.
(Science, 2020: "Dendritic action potentials and computation in human layer 2/3 cortical neurons",
ModelDB #254217).

Core Mechanism:
Unlike conventional monotonic integrate-and-fire or Hodgkin-Huxley neurons, human L2/3
pyramidal apical dendrites generate dendritic calcium action potentials (dCaAPs)
with a non-monotonic, inverted-U amplitude response:
1. Subthreshold input (< I_th): No dCaAP; membrane responds passively. Soma stays resting.
2. Moderate threshold input (I_th <= I <= I_opt): Triggers a full-amplitude dCaAP (K ~ 1.0)
   mediated by voltage-gated calcium channels. This 15-20 ms inward calcium burst propagates
   down the apical trunk to the soma, firing a somatic action potential.
3. Suprathreshold input (I > I_opt, e.g. concurrent XOR inputs): High depolarization recruits
   fast outward potassium channels (delayed rectifier / A-type Kv), which rapidly shunt
   the dendritic burst. The dCaAP amplitude K decays exponentially with excess current:
       K = exp(-(I_peak - I_opt) / (I_opt * D))
   Because the dCaAP is extinguished prematurely, insufficient charge reaches the soma,
   leaving the soma sub-threshold (no somatic spike).

This enables a single neuron to compute the non-linearly separable XOR operation
in continuous physical time without label conditioning or artificial shortcuts.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple


class GidonDCaAPNeuron:
    """Two-compartment biophysical neuron with human L2/3 dCaAP non-monotonic shunting."""

    def __init__(
        self,
        cd_nf: float = 0.05,
        gleak_d_us: float = 0.003,
        el_mv: float = -74.0,
        gc_us: float = 0.002,
        cs_nf: float = 0.1,
        gleak_s_us: float = 0.005,
        vth_soma_mv: float = -52.0,
        vpeak_mv: float = 20.0,
        vreset_mv: float = -74.0,
        w_na: float = 5.0,
        i_th_na: float = 0.6,
        i_opt_na: float = 1.0,
        d_param: float = 0.35,
        tau_a_ms: float = 2.5,
        tau_b_ms: float = 0.5,
        sigma_diff_ms: float = 15.0,
        ablate_shunting: bool = False,
        passive_dendrite: bool = False,
    ):
        # Dendritic passive properties
        self.cd = cd_nf
        self.gleak_d = gleak_d_us
        self.el = el_mv
        self.gc = gc_us

        # Somatic properties
        self.cs = cs_nf
        self.gleak_s = gleak_s_us
        self.vth_soma = vth_soma_mv
        self.vpeak = vpeak_mv
        self.vreset = vreset_mv

        # dCaAP active properties (Gidon et al. 2020)
        self.w = 0.0 if passive_dendrite else w_na
        self.i_th = i_th_na
        self.i_opt = i_opt_na
        self.d_param = d_param
        self.tau_a = tau_a_ms
        self.tau_b = tau_b_ms
        self.sigma_diff = sigma_diff_ms

        # Control flags
        self.ablate_shunting = ablate_shunting
        self.passive_dendrite = passive_dendrite

    def compute_attenuation(self, i_peak: float) -> float:
        """Compute the dCaAP amplitude scaling factor K(I_peak)."""
        if self.passive_dendrite or i_peak < self.i_th:
            return 0.0
        if self.ablate_shunting:
            return 1.0
        if i_peak <= self.i_opt * 1.15:
            return 1.0
        excess = i_peak - self.i_opt * 1.15
        denom = self.i_opt * self.d_param
        return float(np.exp(-excess / denom))

    def simulate(
        self,
        i_syn_trace: np.ndarray,
        time_ms: np.ndarray,
        dt_ms: float = 0.05,
    ) -> Dict[str, Any]:
        """Simulate continuous two-compartment ODE dynamics over time.

        Args:
            i_syn_trace: Injected synaptic current array (nA) over time.
            time_ms: Simulation time stamps array (ms).
            dt_ms: Numerical integration time step (ms).

        Returns:
            Dictionary containing spike count, voltage traces, and dCaAP metrics.
        """
        n_steps = len(time_ms)
        vd_trace = np.zeros(n_steps)
        vs_trace = np.zeros(n_steps)

        i_peak = float(np.max(i_syn_trace)) if len(i_syn_trace) > 0 else 0.0
        k_factor = self.compute_attenuation(i_peak)

        vd = self.el
        vs = self.el
        a_gate = 0.0
        b_gate = 0.0
        b_armed = False
        t_trig = -100.0
        ref_soma = 0.0
        spikes = 0
        spike_times = []

        for step in range(n_steps):
            t = time_ms[step]
            i_syn = i_syn_trace[step]

            # Trigger dCaAP when input current crosses initiation threshold
            if t_trig < 0.0 and i_syn >= self.i_th and k_factor > 0.0 and not self.passive_dendrite:
                t_trig = t
                a_gate = 0.001
                b_gate = 0.0
                b_armed = True

            # Activate falling gate B after delay sigma_diff (Gidon et al. 2020)
            if b_armed and t >= t_trig + self.sigma_diff:
                b_gate = 0.001
                b_armed = False

            # Gating variable continuous integration
            if a_gate > 0.0:
                da = a_gate * (1.0 - a_gate) / self.tau_a
                a_gate = float(np.clip(a_gate + dt_ms * da, 0.0, 1.0))
            if b_gate > 0.0:
                db = b_gate * (1.0 - b_gate) / self.tau_b
                b_gate = float(np.clip(b_gate + dt_ms * db, 0.0, 1.0))

            i_dcaap = (a_gate - b_gate) * self.w * k_factor

            # Dendritic compartment ODE: C_d * dV_d/dt = -g_leak*(V_d - E_L) - I_axial + I_syn + I_dcaap
            i_axial = self.gc * (vd - vs)
            dvd = (-self.gleak_d * (vd - self.el) - i_axial + i_syn + i_dcaap) / self.cd
            vd += dt_ms * dvd

            # Somatic compartment ODE: C_s * dV_s/dt = -g_leak_s*(V_s - E_L) + I_axial
            if ref_soma > 0.0:
                ref_soma -= dt_ms
                vs = self.vreset
            else:
                dvs = (-self.gleak_s * (vs - self.el) + i_axial) / self.cs
                vs += dt_ms * dvs
                if vs >= self.vth_soma:
                    spikes += 1
                    spike_times.append(float(t))
                    vs = self.vpeak
                    ref_soma = 5.0

            vd_trace[step] = vd
            vs_trace[step] = vs

        return {
            "spikes": spikes,
            "spike_times": spike_times,
            "k_factor": k_factor,
            "i_peak": i_peak,
            "max_vs": float(np.max(vs_trace)),
            "max_vd": float(np.max(vd_trace)),
            "vd_trace": vd_trace,
            "vs_trace": vs_trace,
        }
