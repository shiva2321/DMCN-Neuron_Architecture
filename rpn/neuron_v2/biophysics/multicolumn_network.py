"""Multi-Column Cortical Network for Compositional Generalization.

Biophysical architecture:
- Layer 4 (Granular): Feedforward sensory thalamic input and filtering.
- Layer 5 (Infragranular): Auto-associative memory attractor and prototype restoration.
- Layer 2/3 (Supragranular): Long-range horizontal consensus and dynamic compositional binding
  (using circular convolution / Holographic Reduced Representation algebra).
- Layer 6 (Corticothalamic): Feedback confidence gating and gain control.

Each cortical column specializes in a distinct cognitive domain (e.g., entity space vs.
relational space). Unseen compositional combinations are dynamically bound through
inter-column Layer 2/3 horizontal projections without requiring prior co-occurrence training.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute circular convolution of two 1D real vectors via FFT.

    In Holographic Reduced Representations (HRR) and Vector Symbolic Architectures (VSA),
    circular convolution serves as the canonical algebraic binding operator:
    y = a (*) b = ifft(fft(a) * fft(b)).
    """
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def circular_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute circular correlation (approximate unbinding) of two vectors.

    y = a (#) b = ifft(fft(a) * conj(fft(b))).
    """
    return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))


class CorticalColumnModule:
    """A modular cortical column with functional laminar layers (L4, L5, L2/3, L6)."""

    def __init__(
        self,
        dim: int,
        alpha_l5: float = 0.65,
    ) -> None:
        """Initialize cortical column.

        Args:
            dim: Representation vector dimension D.
            alpha_l5: Weight of L5 auto-associative prior vs L4 sensory feedforward drive.
        """
        self.dim = dim
        self.alpha_l5 = alpha_l5

        # Layer 5 auto-associative memory matrix (Hopfield / Hebbian attractor)
        self.memory_prototypes: List[np.ndarray] = []
        self.W_l5: np.ndarray = np.zeros((dim, dim), dtype=np.float64)

        # Laminar state registers (updated dynamically during processing)
        self.l4_state: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.l5_state: np.ndarray = np.zeros(dim, dtype=np.float64)

    def store_prototypes(self, prototypes: np.ndarray) -> None:
        """Store canonical domain prototypes in Layer 5 associative memory.

        Args:
            prototypes: (N, D) array of normalized prototype vectors.
        """
        self.memory_prototypes = [p / (np.linalg.norm(p) + 1e-12) for p in prototypes]

        # Outer-product auto-associative weight matrix (subspace projector)
        W = np.zeros((self.dim, self.dim), dtype=np.float64)
        for p in self.memory_prototypes:
            W += np.outer(p, p)
        self.W_l5 = W

    def process_sensory_input(self, x: np.ndarray) -> np.ndarray:
        """Process incoming feedforward input through Layer 4 and Layer 5.

        Args:
            x: Input vector (dim,).

        Returns:
            Resolved column state z_col (dim,).
        """
        # Layer 4: Sensory feedforward normalization
        norm_x = np.linalg.norm(x)
        if norm_x > 1e-12:
            self.l4_state = x / norm_x
        else:
            self.l4_state = np.zeros_like(x)

        # Layer 5: Attractor clean-up / prior retrieval
        raw_l5 = self.W_l5 @ self.l4_state
        norm_l5 = np.linalg.norm(raw_l5)
        if norm_l5 > 1e-12:
            self.l5_state = raw_l5 / norm_l5
        else:
            self.l5_state = self.l4_state.copy()

        # Integrated column state: blend L4 sensory drive with L5 recurrent prior
        z = (1.0 - self.alpha_l5) * self.l4_state + self.alpha_l5 * self.l5_state
        norm_z = np.linalg.norm(z)
        if norm_z > 1e-12:
            z = z / norm_z
        return z

    def identify_prototype(self, z: np.ndarray) -> Tuple[int, float]:
        """Classify current state against stored domain prototypes.

        Args:
            z: Column state vector (dim,).

        Returns:
            Tuple of (best_prototype_index, cosine_similarity).
        """
        if not self.memory_prototypes:
            return -1, 0.0
        sims = [float(np.dot(z, p)) for p in self.memory_prototypes]
        best_idx = int(np.argmax(sims))
        return best_idx, sims[best_idx]


class MultiColumnCompositionalNetwork:
    """Network of cortical columns communicating via Layer 2/3 horizontal consensus."""

    def __init__(self, dim: int = 256) -> None:
        """Initialize multi-column network.

        Args:
            dim: Dimension of representation vectors.
        """
        self.dim = dim
        self.col_entity = CorticalColumnModule(dim, alpha_l5=0.60)
        self.col_relation = CorticalColumnModule(dim, alpha_l5=0.60)

    def train_columns(
        self,
        entity_prototypes: np.ndarray,
        relation_prototypes: np.ndarray,
    ) -> None:
        """Train domain representations in the respective cortical columns."""
        self.col_entity.store_prototypes(entity_prototypes)
        self.col_relation.store_prototypes(relation_prototypes)

    def forward(
        self,
        entity_input: np.ndarray,
        relation_input: np.ndarray,
        decouple_horizontal: bool = False,
        shuffle_binding: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """Perform forward multi-column compositional binding.

        Args:
            entity_input: Sensory input to entity column (dim,).
            relation_input: Sensory input to relation column (dim,).
            decouple_horizontal: Null control; severs inter-column L2/3 consensus.
            shuffle_binding: Null control; permutes horizontal projections.
            rng: RandomState for null controls if needed.

        Returns:
            Dictionary with bound compositional representation and diagnostic states.
        """
        # Step 1: Local columnar processing in L4 and L5
        z_entity = self.col_entity.process_sensory_input(entity_input)
        z_relation = self.col_relation.process_sensory_input(relation_input)

        # Step 2: Layer 2/3 Supragranular Horizontal Consensus & Binding
        if decouple_horizontal:
            # Control: No communication between columns
            bound_rep = np.zeros(self.dim, dtype=np.float64)
            l6_gain = 0.0
        elif shuffle_binding:
            # Control: Shuffled / scrambled horizontal projection
            local_rng = rng if rng is not None else np.random.RandomState(42)
            perm = local_rng.permutation(self.dim)
            shuffled_rel = z_relation[perm]
            bound_rep = circular_convolution(z_entity, shuffled_rel)
            norm_b = np.linalg.norm(bound_rep)
            if norm_b > 1e-12:
                bound_rep = bound_rep / norm_b
            l6_gain = 0.5
        else:
            # Active biophysical data path: Layer 2/3 circular convolution binding
            bound_rep = circular_convolution(z_entity, z_relation)
            norm_b = np.linalg.norm(bound_rep)
            if norm_b > 1e-12:
                bound_rep = bound_rep / norm_b

            # Layer 6 corticothalamic confidence gain
            # Product of columnar recognition fidelities
            _, sim_e = self.col_entity.identify_prototype(z_entity)
            _, sim_r = self.col_relation.identify_prototype(z_relation)
            l6_gain = float(np.clip(sim_e * sim_r, 0.0, 1.0))

        # Modulate output by L6 confidence gain
        y_out = bound_rep * l6_gain

        return {
            "bound_rep": bound_rep,
            "y_out": y_out,
            "z_entity": z_entity,
            "z_relation": z_relation,
            "l6_gain": np.array([l6_gain]),
        }

    def predict_compositional_class(
        self,
        entity_input: np.ndarray,
        relation_input: np.ndarray,
        candidate_codebook: np.ndarray,
        decouple_horizontal: bool = False,
        shuffle_binding: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[int, float]:
        """Classify compositional query against a codebook of all 128 compositional targets.

        Args:
            entity_input: Sensory input vector.
            relation_input: Sensory relation vector.
            candidate_codebook: (128, dim) array of target bound representations.
            decouple_horizontal: Null control flag.
            shuffle_binding: Null control flag.
            rng: RandomState for null controls.

        Returns:
            Tuple of (predicted_class_index, cosine_similarity).
        """
        out = self.forward(
            entity_input,
            relation_input,
            decouple_horizontal=decouple_horizontal,
            shuffle_binding=shuffle_binding,
            rng=rng,
        )
        y = out["bound_rep"]
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-12:
            return -1, 0.0

        # Cosine similarity against candidate codebook
        # candidate_codebook is assumed normalized
        sims = candidate_codebook @ y
        best_idx = int(np.argmax(sims))
        return best_idx, float(sims[best_idx])


class MonolithicFlatMemoryBaseline:
    """Non-factorized associative memory baseline that memorizes training pairs directly."""

    def __init__(self, dim: int = 256) -> None:
        """Initialize monolithic flat memory."""
        self.dim = dim
        self.stored_keys: List[np.ndarray] = []
        self.stored_targets: List[np.ndarray] = []
        self.stored_class_indices: List[int] = []

    def fit(
        self,
        train_entity_vectors: np.ndarray,
        train_relation_vectors: np.ndarray,
        train_class_indices: List[int],
        train_targets: np.ndarray,
    ) -> None:
        """Memorize training pairs as joint flat concatenated vectors [e; r]."""
        self.stored_keys = []
        self.stored_targets = []
        self.stored_class_indices = list(train_class_indices)
        for i in range(len(train_class_indices)):
            e = train_entity_vectors[i]
            r = train_relation_vectors[i]
            flat_key = np.concatenate([e, r])
            norm_k = np.linalg.norm(flat_key)
            if norm_k > 1e-12:
                flat_key = flat_key / norm_k
            self.stored_keys.append(flat_key)
            self.stored_targets.append(train_targets[i])

    def query(
        self,
        entity_input: np.ndarray,
        relation_input: np.ndarray,
    ) -> Tuple[int, float]:
        """Query flat memory with input vectors.

        For unseen combinations not in the training set, nearest-neighbor
        matching on [e; r] will select the nearest seen training pair,
        failing compositional generalization.
        """
        if not self.stored_keys:
            return -1, 0.0
        query_key = np.concatenate([entity_input, relation_input])
        norm_q = np.linalg.norm(query_key)
        if norm_q > 1e-12:
            query_key = query_key / norm_q

        sims = [float(np.dot(query_key, k)) for k in self.stored_keys]
        best_local_idx = int(np.argmax(sims))
        predicted_class = self.stored_class_indices[best_local_idx]
        return predicted_class, sims[best_local_idx]
