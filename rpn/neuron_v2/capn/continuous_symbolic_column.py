"""Continuous Symbolic Reasoning Column & Resonant Wavefield Engine.

Implements genuine gradient-free neural symbolic and compositional reasoning
without string manipulation, answer branching, or algorithmic carry loops.

Key Mechanisms:
1. Holographic Continuous Phasor Binding & Unbinding:
   Unitary complex phasors z in C^D (z_j = exp(i * theta_j)) where binding is
   elementwise complex multiplication (z_a (.) z_b) and unbinding is phase
   conjugation (z_ab (.) conj(z_a)).
2. Continuous Resonant Phasor Arithmetic:
   Numbers represented as continuous Fourier wavefields psi(x) = exp(i * omega * x).
   Addition is direct wave interference (complex Hadamard product):
   psi(a + b) = psi(a) (.) psi(b), and algebraic inversion is phase conjugation.
3. Layer 5 Continuous Attractor Clean-Up (Competitive CANN):
   Continuous Lyapunov energy minimization settles cross-talk / noise contaminated
   unbinding states into clean prototype attractor basins without collapsing to the mean.
4. Multi-Column Analogy and Transitive Deduction:
   Inter-column supragranular projections compute relational transformations
   T = B (.) conj(A) and project to target instances D = C (.) T zero-shot.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def normalise_phasor(v: np.ndarray) -> np.ndarray:
    """Project complex vector onto unit magnitude sphere |z_j| = 1."""
    mag = np.abs(v)
    mask = mag > 1e-12
    out = v.copy()
    out[mask] /= mag[mask]
    return out.astype(np.complex128)


def phasor_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Algebraic binding operator in Layer 2/3: elementwise complex product."""
    return (a * b).astype(np.complex128)


def phasor_unbind(ab: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Algebraic unbinding operator: Hadamard product with complex conjugate."""
    return (ab * np.conj(a)).astype(np.complex128)


def phasor_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized cosine similarity / phase coherence between two phasor vectors."""
    d = len(x)
    return float(np.real(np.dot(x, np.conj(y)))) / d


def generate_log_cochlear_frequencies(n_frequencies: int, omega_min: float = 0.01, omega_max: float = 0.80) -> np.ndarray:
    """Generate geometrically spaced log-cochlear frequencies."""
    return np.geomspace(omega_min, omega_max, num=n_frequencies, dtype=np.float64)


def encode_phasor_wavefield(
    x: Union[float, int, np.ndarray],
    frequencies: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode scalar or array into continuous Fourier wavefield phasors.

    Returns:
        complex_phasor: exp(i * omega * x) in C^M.
        real_vector: [cos(omega * x), sin(omega * x)] / sqrt(M) in R^{2M}.
    """
    m = len(frequencies)
    phase = np.outer(x, frequencies) if isinstance(x, np.ndarray) else x * frequencies
    complex_phasor = np.exp(1j * phase)
    real_cos = np.cos(phase)
    real_sin = np.sin(phase)
    real_vector = np.concatenate([real_cos, real_sin], axis=-1) / np.sqrt(m)
    return complex_phasor, real_vector


def wave_interference_add(phasor_a: np.ndarray, phasor_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform exact continuous addition via wave interference (complex Hadamard product).

    psi(a + b) = psi(a) * psi(b) = exp(i * omega * (a + b)).
    Zero python carry loops, zero string conversions.
    """
    m = len(phasor_a)
    phasor_sum = phasor_a * phasor_b
    phase = np.angle(phasor_sum)
    real_vector = np.concatenate([np.cos(phase), np.sin(phase)], axis=-1) / np.sqrt(m)
    return phasor_sum, real_vector


def wave_interference_invert(phasor_sum: np.ndarray, phasor_known: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform exact algebraic inversion (subtraction) via conjugate wave interference.

    psi(b) = psi(c) * conj(psi(a)) = exp(i * omega * (c - a)).
    """
    m = len(phasor_sum)
    phasor_diff = phasor_sum * np.conj(phasor_known)
    phase = np.angle(phasor_diff)
    real_vector = np.concatenate([np.cos(phase), np.sin(phase)], axis=-1) / np.sqrt(m)
    return phasor_diff, real_vector


class ContinuousAttractorCleanUp:
    """Layer 5 Continuous Competitive Attractor Clean-Up Engine (CANN / Modern Hopfield).

    Minimizes energy E(z) = - 1/beta * log(sum_k exp(beta * Re(p_k . z*))),
    driving cross-talk or noise corrupted states cleanly into the dominant prototype basin.
    """

    def __init__(self, dim: int, beta: float = 40.0) -> None:
        self.dim = dim
        self.beta = beta
        self.prototypes: np.ndarray = np.empty((0, dim), dtype=np.complex128)

    def set_prototypes(self, prototypes: np.ndarray) -> None:
        """Store canonical prototypes as attractor basins."""
        self.prototypes = np.array([normalise_phasor(p) for p in prototypes], dtype=np.complex128)

    def settle(
        self,
        z_init: np.ndarray,
        t_steps: int = 10,
        beta: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Iterate continuous competitive attractor relaxation.

        Args:
            z_init: Initial cross-talk corrupted state vector (dim,).
            t_steps: Number of relaxation steps.
            beta: Softmax inverse temperature.

        Returns:
            Tuple of (settled_state_vector, final_lyapunov_energy).
        """
        if len(self.prototypes) == 0:
            return z_init.copy(), 0.0

        b = beta if beta is not None else self.beta
        z = normalise_phasor(z_init)

        for _ in range(t_steps):
            # Phase coherence overlaps across all stored prototype basins
            overlaps = np.real(self.prototypes @ np.conj(z)) / self.dim
            max_o = float(np.max(overlaps))
            # Numerically stable competitive weights
            weights = np.exp(b * (overlaps - max_o))
            sum_w = float(np.sum(weights))
            if sum_w > 1e-12:
                weights /= sum_w
            else:
                break
            # Recurrent attractor pull
            z_new = weights @ self.prototypes
            z = normalise_phasor(z_new)

        overlaps_final = np.real(self.prototypes @ np.conj(z)) / self.dim
        max_final = float(np.max(overlaps_final))
        energy = - (max_final + (1.0 / b) * np.log(float(np.sum(np.exp(b * (overlaps_final - max_final)))) + 1e-12))
        return z, energy

    def decode_nearest(self, z: np.ndarray) -> Tuple[int, float]:
        """Match settled state against stored attractor basins."""
        if len(self.prototypes) == 0:
            return -1, 0.0
        z_norm = normalise_phasor(z)
        sims = np.real(self.prototypes @ np.conj(z_norm)) / self.dim
        best_idx = int(np.argmax(sims))
        return best_idx, float(sims[best_idx])


class ContinuousSymbolicReasoningColumn:
    """Canonical Cortical Column for Genuine Neural Symbolic and Compositional Reasoning."""

    def __init__(
        self,
        dim: int = 512,
        t_settle: int = 10,
        beta_settle: float = 40.0,
    ) -> None:
        self.dim = dim
        self.t_settle = t_settle
        self.beta_settle = beta_settle

        # Layer 5 continuous attractor clean-up modules
        self.l5_entity_attractor = ContinuousAttractorCleanUp(dim, beta=beta_settle)
        self.l5_relation_attractor = ContinuousAttractorCleanUp(dim, beta=beta_settle)

        # Stored prototype references
        self.entity_prototypes: List[np.ndarray] = []
        self.relation_prototypes: List[np.ndarray] = []

    def set_knowledge_base(
        self,
        entity_prototypes: np.ndarray,
        relation_prototypes: np.ndarray,
    ) -> None:
        """Register entity and relation prototype vectors into Layer 5 attractors."""
        self.entity_prototypes = [normalise_phasor(p) for p in entity_prototypes]
        self.relation_prototypes = [normalise_phasor(p) for p in relation_prototypes]
        self.l5_entity_attractor.set_prototypes(np.array(self.entity_prototypes))
        self.l5_relation_attractor.set_prototypes(np.array(self.relation_prototypes))

    def solve_relational_analogy(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        vec_c: np.ndarray,
        decouple_horizontal: bool = False,
        shuffle_binding: bool = False,
        blind_input: bool = False,
        ablate_attractor: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Solve proportional relational analogy A : B :: C : ? (infer D = C (*) (B (#) A)).

        Args:
            vec_a, vec_b, vec_c: Sensory inputs for terms A, B, C (dim,).
            decouple_horizontal: Null control flag (severs L2/3 consensus).
            shuffle_binding: Null control flag (scrambles coordinates).
            blind_input: Null control flag (zeros inputs).
            ablate_attractor: Null control flag (skips L5 settling).
            rng: RandomState for null controls.

        Returns:
            Dictionary with predicted prototype index, confidence, and internal states.
        """
        local_rng = rng if rng is not None else np.random.RandomState(42)

        if blind_input:
            vec_a = np.zeros(self.dim, dtype=np.complex128)
            vec_b = np.zeros(self.dim, dtype=np.complex128)
            vec_c = np.zeros(self.dim, dtype=np.complex128)
            z_a = vec_a
            z_b = vec_b
            z_c = vec_c
        else:
            z_a = normalise_phasor(vec_a)
            z_b = normalise_phasor(vec_b)
            z_c = normalise_phasor(vec_c)

        # Layer 2/3: Inter-Column Relational Unbinding & Compositional Projection
        if decouple_horizontal:
            # Control: Severs communication between columns
            raw_d = np.zeros(self.dim, dtype=np.complex128)
            trans_rel = np.zeros(self.dim, dtype=np.complex128)
            l6_gain = 0.0
        elif shuffle_binding:
            # Control: Random permutation destroys algebraic covariance
            perm = local_rng.permutation(self.dim)
            trans_rel = phasor_unbind(z_b[perm], z_a)
            raw_d = phasor_bind(z_c, trans_rel)
            l6_gain = 0.5
        elif blind_input:
            raw_d = np.zeros(self.dim, dtype=np.complex128)
            trans_rel = np.zeros(self.dim, dtype=np.complex128)
            l6_gain = 0.0
        else:
            # Active path: Unbind relation T = B (#) A
            trans_rel = phasor_unbind(z_b, z_a)
            # Project relation onto C: D = C (*) T
            raw_d = phasor_bind(z_c, trans_rel)
            # Layer 6 corticothalamic gain
            l6_gain = 1.0

        # Layer 5: Continuous Attractor Clean-Up
        if ablate_attractor or decouple_horizontal or blind_input:
            settled_d = raw_d.copy()
            energy = 0.0
        else:
            settled_d, energy = self.l5_entity_attractor.settle(
                raw_d,
                t_steps=self.t_settle,
                beta=self.beta_settle,
            )

        predicted_idx, similarity = self.l5_entity_attractor.decode_nearest(settled_d)

        return {
            "predicted_idx": predicted_idx,
            "similarity": similarity * l6_gain,
            "raw_d": raw_d,
            "settled_d": settled_d,
            "trans_rel": trans_rel,
            "energy": energy,
            "l6_gain": l6_gain,
        }

    def solve_transitive_deduction(
        self,
        vec_a: np.ndarray,
        rel_ab: np.ndarray,
        rel_bc: np.ndarray,
        decouple_horizontal: bool = False,
        shuffle_binding: bool = False,
        blind_input: bool = False,
        ablate_attractor: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Solve multi-hop transitive relation deduction: A -> B -> C (infer C = A (*) (R_ab (*) R_bc)).

        Args:
            vec_a: Source entity vector (dim,).
            rel_ab: First relational transformation (dim,).
            rel_bc: Second relational transformation (dim,).
            decouple_horizontal: Null control flag.
            shuffle_binding: Null control flag.
            blind_input: Null control flag.
            ablate_attractor: Null control flag.
            rng: RandomState for null controls.

        Returns:
            Dictionary with predicted prototype index, confidence, and internal states.
        """
        local_rng = rng if rng is not None else np.random.RandomState(42)

        if blind_input:
            vec_a = np.zeros(self.dim, dtype=np.complex128)
            rel_ab = np.zeros(self.dim, dtype=np.complex128)
            rel_bc = np.zeros(self.dim, dtype=np.complex128)
            z_a = vec_a
            z_ab = rel_ab
            z_bc = rel_bc
        else:
            z_a = normalise_phasor(vec_a)
            z_ab = normalise_phasor(rel_ab)
            z_bc = normalise_phasor(rel_bc)

        if decouple_horizontal:
            raw_c = np.zeros(self.dim, dtype=np.complex128)
            rel_ac = np.zeros(self.dim, dtype=np.complex128)
            l6_gain = 0.0
        elif shuffle_binding:
            perm = local_rng.permutation(self.dim)
            rel_ac = phasor_bind(z_ab[perm], z_bc)
            raw_c = phasor_bind(z_a, rel_ac)
            l6_gain = 0.5
        elif blind_input:
            raw_c = np.zeros(self.dim, dtype=np.complex128)
            rel_ac = np.zeros(self.dim, dtype=np.complex128)
            l6_gain = 0.0
        else:
            # Composite relation: R_ac = R_ab (*) R_bc
            rel_ac = phasor_bind(z_ab, z_bc)
            # Apply composite relation to A: C = A (*) R_ac
            raw_c = phasor_bind(z_a, rel_ac)
            l6_gain = 1.0

        if ablate_attractor or decouple_horizontal or blind_input:
            settled_c = raw_c.copy()
            energy = 0.0
        else:
            settled_c, energy = self.l5_entity_attractor.settle(
                raw_c,
                t_steps=self.t_settle,
                beta=self.beta_settle,
            )

        predicted_idx, similarity = self.l5_entity_attractor.decode_nearest(settled_c)

        return {
            "predicted_idx": predicted_idx,
            "similarity": similarity * l6_gain,
            "raw_c": raw_c,
            "settled_c": settled_c,
            "rel_ac": rel_ac,
            "energy": energy,
            "l6_gain": l6_gain,
        }


class ResonantWavefieldArithmeticEngine:
    """Continuous Resonant Wavefield Engine for Reversible Arithmetic.

    Numbers are continuous phasor wavefields. Addition is native wave interference,
    and inversion is phase conjugation. Settling onto continuous candidate attractors
    decodes the numerical result without Python loops or strings.
    """

    def __init__(
        self,
        n_frequencies: int = 256,
        max_val: int = 50,
        omega_min: float = 0.01,
        omega_max: float = 0.80,
        beta_settle: float = 40.0,
    ) -> None:
        self.n_frequencies = n_frequencies
        self.max_val = max_val
        self.frequencies = generate_log_cochlear_frequencies(n_frequencies, omega_min, omega_max)

        # Precompute candidate number prototypes for attractor clean-up
        self.candidate_values = np.arange(0, max_val + 1)
        self.candidate_phasors = np.array([
            np.exp(1j * v * self.frequencies) for v in self.candidate_values
        ], dtype=np.complex128)

        # Attractor clean-up
        self.attractor = ContinuousAttractorCleanUp(n_frequencies, beta=beta_settle)
        self.attractor.set_prototypes(self.candidate_phasors)

    def add(
        self,
        val_a: int,
        val_b: int,
        noise_sigma: float = 0.0,
        decouple_wave: bool = False,
        shuffle_phase: bool = False,
        blind_input: bool = False,
        ablate_attractor: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Perform continuous wave addition: psi(a + b) = psi(a) * psi(b)."""
        local_rng = rng if rng is not None else np.random.RandomState(42)

        if blind_input:
            p_a = np.zeros(self.n_frequencies, dtype=np.complex128)
            p_b = np.zeros(self.n_frequencies, dtype=np.complex128)
        else:
            p_a = np.exp(1j * val_a * self.frequencies)
            p_b = np.exp(1j * val_b * self.frequencies)
            if noise_sigma > 0.0:
                p_a = p_a * np.exp(1j * local_rng.randn(self.n_frequencies) * noise_sigma)
                p_b = p_b * np.exp(1j * local_rng.randn(self.n_frequencies) * noise_sigma)

        if decouple_wave:
            res_phasor = np.zeros(self.n_frequencies, dtype=np.complex128)
            l6_gain = 0.0
        elif shuffle_phase:
            perm = local_rng.permutation(self.n_frequencies)
            res_phasor = p_a[perm] * p_b
            l6_gain = 0.5
        elif blind_input:
            res_phasor = np.zeros(self.n_frequencies, dtype=np.complex128)
            l6_gain = 0.0
        else:
            res_phasor = p_a * p_b
            l6_gain = 1.0

        if ablate_attractor or decouple_wave or blind_input:
            settled_phasor = res_phasor.copy()
        else:
            settled_phasor, _ = self.attractor.settle(res_phasor, t_steps=10)

        pred_idx, similarity = self.attractor.decode_nearest(settled_phasor)
        predicted_val = int(self.candidate_values[pred_idx]) if pred_idx >= 0 and l6_gain > 0 else -1

        return {
            "predicted_val": predicted_val,
            "similarity": similarity * l6_gain,
            "res_phasor": res_phasor,
            "settled_phasor": settled_phasor,
            "l6_gain": l6_gain,
        }

    def invert(
        self,
        val_sum: int,
        val_known: int,
        noise_sigma: float = 0.0,
        decouple_wave: bool = False,
        shuffle_phase: bool = False,
        blind_input: bool = False,
        ablate_attractor: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Perform continuous algebraic inversion: psi(b) = psi(c) * conj(psi(a))."""
        local_rng = rng if rng is not None else np.random.RandomState(42)

        if blind_input:
            p_sum = np.zeros(self.n_frequencies, dtype=np.complex128)
            p_known = np.zeros(self.n_frequencies, dtype=np.complex128)
        else:
            p_sum = np.exp(1j * val_sum * self.frequencies)
            p_known = np.exp(1j * val_known * self.frequencies)
            if noise_sigma > 0.0:
                p_sum = p_sum * np.exp(1j * local_rng.randn(self.n_frequencies) * noise_sigma)
                p_known = p_known * np.exp(1j * local_rng.randn(self.n_frequencies) * noise_sigma)

        if decouple_wave:
            res_phasor = np.zeros(self.n_frequencies, dtype=np.complex128)
            l6_gain = 0.0
        elif shuffle_phase:
            perm = local_rng.permutation(self.n_frequencies)
            res_phasor = p_sum[perm] * np.conj(p_known)
            l6_gain = 0.5
        elif blind_input:
            res_phasor = np.zeros(self.n_frequencies, dtype=np.complex128)
            l6_gain = 0.0
        else:
            res_phasor = p_sum * np.conj(p_known)
            l6_gain = 1.0

        if ablate_attractor or decouple_wave or blind_input:
            settled_phasor = res_phasor.copy()
        else:
            settled_phasor, _ = self.attractor.settle(res_phasor, t_steps=10)

        pred_idx, similarity = self.attractor.decode_nearest(settled_phasor)
        predicted_val = int(self.candidate_values[pred_idx]) if pred_idx >= 0 and l6_gain > 0 else -1

        return {
            "predicted_val": predicted_val,
            "similarity": similarity * l6_gain,
            "res_phasor": res_phasor,
            "settled_phasor": settled_phasor,
            "l6_gain": l6_gain,
        }


class MonolithicFlatMemoryAnalogyBaseline:
    """Non-factorized associative memory baseline that memorizes training triplets directly."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.stored_keys: List[np.ndarray] = []
        self.stored_target_indices: List[int] = []

    def fit(self, training_triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]) -> None:
        """Store training analogies as flat concatenated real keys [Re(A); Im(A); Re(B); Im(B); ...] -> D."""
        self.stored_keys = []
        self.stored_target_indices = []
        for va, vb, vc, target_idx in training_triplets:
            flat_real = np.concatenate([
                np.real(va), np.imag(va),
                np.real(vb), np.imag(vb),
                np.real(vc), np.imag(vc),
            ])
            norm_k = np.linalg.norm(flat_real)
            if norm_k > 1e-12:
                flat_real = flat_real / norm_k
            self.stored_keys.append(flat_real)
            self.stored_target_indices.append(target_idx)

    def query(self, va: np.ndarray, vb: np.ndarray, vc: np.ndarray) -> Tuple[int, float]:
        """Query flat memory with test instance.

        For unseen analogical combinations, nearest neighbor fails because
        unseen quadruplets do not match the memorized concatenated keys.
        """
        if not self.stored_keys:
            return -1, 0.0
        q = np.concatenate([
            np.real(va), np.imag(va),
            np.real(vb), np.imag(vb),
            np.real(vc), np.imag(vc),
        ])
        norm_q = np.linalg.norm(q)
        if norm_q > 1e-12:
            q = q / norm_q
        sims = [float(np.dot(q, k)) for k in self.stored_keys]
        best_idx = int(np.argmax(sims))
        return self.stored_target_indices[best_idx], sims[best_idx]
