"""Baselines for shootout: RXCA-style additive hologram, MLP, RPN column."""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from neuron_v2.core.born_gate import rectified_born
from neuron_v2.core.delta_memory import DeltaMemory
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpn import RPN
from neuron_v2.substrate.phasor import bind, bundle, normalise, permute, random_hv, sim_batch, unbind


def _role_seed(name: str) -> int:
    """Deterministic seed for compositional role HVs (matches CompositionalColumn)."""
    digest = hashlib.md5(f"compositional-role:{name}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


class RoleBoundCompositeCtx:
    """Role-bound composite context: bundle(bind(ρ¹ role_color, color), bind(ρ² role_shape, shape))."""

    def __init__(self, D: int, items: ItemTable):
        self.D = D
        self.items = items
        self.role_color = random_hv(D, seed=_role_seed("role_color"))
        self.role_shape = random_hv(D, seed=_role_seed("role_shape"))

    def __call__(self, color: str, shape: str) -> np.ndarray:
        color_ctx = bind(permute(self.role_color, 1), self.items[color])
        shape_ctx = bind(permute(self.role_shape, 2), self.items[shape])
        return bundle(color_ctx, shape_ctx)


class HolographicCompositeBrain:
    """RXCA-style hologram on role-bound composite contexts (not token ctx keys)."""

    def __init__(self, D: int, items: ItemTable):
        self.D = D
        self.items = items
        self.ctx_builder = RoleBoundCompositeCtx(D, items)
        self.surface = np.zeros(D, dtype=np.complex64)
        self.n_writes = 0

    def learn(self, color: str, shape: str, object_token: str) -> None:
        ctx = self.ctx_builder(color, shape)
        self.surface = (
            self.surface + bind(ctx, self.items[object_token])
        ).astype(np.complex64)
        self.n_writes += 1

    def read(
        self,
        color: str,
        shape: str,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
    ) -> tuple[Optional[str], float]:
        ctx = self.ctx_builder(color, shape)
        pred = unbind(self.surface, ctx)
        return _read_from_echo(
            self.items,
            pred,
            gate_threshold,
            confidence_threshold,
            min_resonance,
            float(np.linalg.norm(pred)),
            self.D,
        )


class SingleRPNComposite:
    """One delta RPN memory keyed by role-bound composite HV (no column routing)."""

    def __init__(self, D: int, items: ItemTable, mode: str = "delta"):
        self.D = D
        self.items = items
        self.ctx_builder = RoleBoundCompositeCtx(D, items)
        self.mem = DeltaMemory(D, mode=mode)

    def learn(self, color: str, shape: str, object_token: str) -> None:
        self.mem.write(self.ctx_builder(color, shape), self.items[object_token])

    def read(
        self,
        color: str,
        shape: str,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
    ) -> tuple[Optional[str], float]:
        pred = self.mem.predict(self.ctx_builder(color, shape))
        return _read_from_echo(
            self.items,
            pred,
            gate_threshold,
            confidence_threshold,
            min_resonance,
            float(np.linalg.norm(pred)),
            self.D,
        )


def _stable_route(token: str, n: int, salt: str = "rpn-col") -> int:
    digest = hashlib.md5(f"{salt}:{token}".encode()).hexdigest()
    return int(digest[:8], 16) % n


def _read_from_echo(
    items: ItemTable,
    echo: np.ndarray,
    gate_threshold: float,
    confidence_threshold: float,
    min_resonance: float,
    pred_norm: float,
    D: int,
) -> tuple[Optional[str], float]:
    confidence = pred_norm / np.sqrt(D)
    if confidence < confidence_threshold:
        return None, 0.0
    echo = normalise(echo)
    tokens, mat = items.matrix()
    res = sim_batch(mat, echo)
    out = rectified_born(
        res,
        gate_threshold=gate_threshold,
        min_resonance=min_resonance,
    )
    if not out.gate_passed:
        return None, out.best_prob
    return tokens[out.best_idx], out.best_prob


class RXCAAdditiveBrain:
    """RXCA PartitionedBrain-style single surface: surface += bind(ctx, target)."""

    def __init__(self, D: int, items: ItemTable):
        self.D = D
        self.items = items
        self.surface = np.zeros(D, dtype=np.complex64)
        self.n_writes = 0

    def learn(self, ctx_token: str, target_token: str) -> None:
        self.surface = (
            self.surface + bind(self.items[ctx_token], self.items[target_token])
        ).astype(np.complex64)
        self.n_writes += 1

    def read(
        self,
        ctx_token: str,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
    ) -> tuple[Optional[str], float]:
        ctx = self.items[ctx_token]
        pred = unbind(self.surface, ctx)
        return _read_from_echo(
            self.items,
            pred,
            gate_threshold,
            confidence_threshold,
            min_resonance,
            float(np.linalg.norm(pred)),
            self.D,
        )


class RPNColumn:
    """Shard associations across N delta RPNs by context hash (specialist column)."""

    def __init__(self, D: int, items: ItemTable, n_neurons: int, mode: str = "delta"):
        if n_neurons < 1:
            raise ValueError("n_neurons must be >= 1")
        self.D = D
        self.items = items
        self.n_neurons = n_neurons
        self.neurons = [RPN(D, items, mode=mode) for _ in range(n_neurons)]

    def _idx(self, ctx_token: str) -> int:
        return _stable_route(ctx_token, self.n_neurons)

    def learn(self, ctx_token: str, target_token: str) -> None:
        self.neurons[self._idx(ctx_token)].learn(ctx_token, target_token)

    def read(
        self,
        ctx_token: str,
        gate_threshold: float = 0.05,
        confidence_threshold: float = 0.1,
        min_resonance: float = 0.1,
    ) -> tuple[Optional[str], float]:
        return self.neurons[self._idx(ctx_token)].read(
            ctx_token,
            gate_threshold=gate_threshold,
            confidence_threshold=confidence_threshold,
            min_resonance=min_resonance,
        )


def _hv_features(hv: np.ndarray) -> np.ndarray:
    return np.concatenate([hv.real, hv.imag]).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


class OnlineMLP:
    """2-layer MLP, one SGD step per (ctx, target) — online continual baseline."""

    def __init__(
        self,
        D: int,
        items: ItemTable,
        hidden: int = 256,
        lr: float = 0.05,
        seed: int = 0,
    ):
        self.D = D
        self.items = items
        self.hidden = hidden
        self.lr = lr
        rng = np.random.RandomState(seed)
        in_dim = 2 * D
        self.W1 = (rng.randn(in_dim, hidden) * 0.02).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.randn(hidden, 0) * 0.02).astype(np.float32)  # resized on first learn
        self.b2 = np.zeros(0, dtype=np.float32)
        self._token_to_idx: dict[str, int] = {}

    def _ensure_output(self, target_token: str) -> int:
        if target_token not in self._token_to_idx:
            idx = len(self._token_to_idx)
            self._token_to_idx[target_token] = idx
            # grow output layer
            rng = np.random.RandomState(idx + 17)
            new_col = (rng.randn(self.hidden, 1) * 0.02).astype(np.float32)
            if self.W2.size == 0:
                self.W2 = new_col
                self.b2 = np.zeros(1, dtype=np.float32)
            else:
                self.W2 = np.concatenate([self.W2, new_col], axis=1)
                self.b2 = np.concatenate([self.b2, np.zeros(1, dtype=np.float32)])
        return self._token_to_idx[target_token]

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return h, logits, _softmax(logits)

    def learn(self, ctx_token: str, target_token: str) -> None:
        y_idx = self._ensure_output(target_token)
        x = _hv_features(self.items[ctx_token])
        h, logits, probs = self._forward(x)
        n = len(probs)
        # cross-entropy gradient at output
        d_logits = probs.copy()
        d_logits[y_idx] -= 1.0
        d_W2 = np.outer(h, d_logits)
        d_b2 = d_logits
        d_h = self.W2 @ d_logits
        d_h[h <= 0] = 0
        d_W1 = np.outer(x, d_h)
        d_b1 = d_h
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

    def read(self, ctx_token: str, **_kwargs) -> tuple[Optional[str], float]:
        if self.W2.size == 0:
            return None, 0.0
        x = _hv_features(self.items[ctx_token])
        _, logits, probs = self._forward(x)
        idx = int(np.argmax(probs))
        inv = {t: i for t, i in self._token_to_idx.items()}
        token = next(t for t, i in inv.items() if i == idx)
        return token, float(probs[idx])


class SingleRPN:
    """Thin wrapper so all baselines share learn/read API."""

    def __init__(self, D: int, items: ItemTable, mode: str = "delta"):
        self.rpn = RPN(D, items, mode=mode)

    def learn(self, ctx_token: str, target_token: str) -> None:
        self.rpn.learn(ctx_token, target_token)

    def read(self, ctx_token: str, **kwargs) -> tuple[Optional[str], float]:
        return self.rpn.read(ctx_token, **kwargs)


class AdditiveMatrixBrain:
    """Same read path as RPN but DeltaMemory in additive mode (matrix outer-product sum)."""

    def __init__(self, D: int, items: ItemTable):
        self.D = D
        self.items = items
        self.mem = DeltaMemory(D, mode="additive")

    def learn(self, ctx_token: str, target_token: str) -> None:
        self.mem.write(self.items[ctx_token], self.items[target_token])

    def read(self, ctx_token: str, **kwargs) -> tuple[Optional[str], float]:
        pred = self.mem.predict(self.items[ctx_token])
        return _read_from_echo(
            self.items,
            pred,
            kwargs.get("gate_threshold", 0.05),
            kwargs.get("confidence_threshold", 0.1),
            kwargs.get("min_resonance", 0.1),
            float(np.linalg.norm(pred)),
            self.D,
        )
