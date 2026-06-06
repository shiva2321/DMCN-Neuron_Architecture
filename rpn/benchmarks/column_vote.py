"""Column vote vs routed shard (B-3).

Compares read policies on the same learned weights:
  - Token assoc: hash-routed vs hash-learn + vote-all-neurons
  - Compositional: color-routed vs color-vote (shape ctx only)
  - Compositional: hash-sharded composite routed vs vote

Run: python -m benchmarks.column_vote
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuron_v2.core.column import ColorShardedColumn, HashShardedCompositeColumn
from neuron_v2.core.item_table import ItemTable
from benchmarks.baselines import RPNColumn


@dataclass
class VoteConfig:
    D: int = 512
    n_neurons: int = 8
    M_token: int = 256
    n_colors: int = 8
    n_shapes: int = 32
    seed: int = 42
    token_vote_min_acc: float = 0.95
    color_route_min_acc: float = 0.95
    hash_route_min_acc: float = 0.90
    color_vote_max_acc: float = 0.50


@dataclass
class VoteResult:
    name: str
    accuracy: float
    scenario: str


def _make_token_pairs(M: int, rng: np.random.RandomState) -> list[tuple[str, str]]:
    ctx_pool = [f"ctx_{i}" for i in range(M * 2)]
    tgt_pool = [f"tgt_{i}" for i in range(M * 2)]
    pairs: list[tuple[str, str]] = []
    used_ctx: set[str] = set()
    while len(pairs) < M:
        c = ctx_pool[int(rng.randint(0, len(ctx_pool)))]
        t = tgt_pool[int(rng.randint(0, len(tgt_pool)))]
        if c in used_ctx:
            continue
        used_ctx.add(c)
        pairs.append((c, t))
    return pairs


def _make_grid(n_colors: int, n_shapes: int) -> list[tuple[str, str, str]]:
    return [
        (f"color_{ci}", f"shape_{si}", f"obj_{ci}_{si}")
        for ci in range(n_colors)
        for si in range(n_shapes)
    ]


def _register_compositional(items: ItemTable, pairs: list[tuple[str, str, str]]) -> None:
    for color, shape, obj in pairs:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]


def _token_accuracy(model, pairs: list[tuple[str, str]]) -> float:
    ok = sum(1 for c, t in pairs if model.read(c)[0] == t)
    return ok / len(pairs)


def _grid_accuracy(model, pairs: list[tuple[str, str, str]]) -> float:
    ok = sum(1 for c, s, o in pairs if model.read(c, s)[0] == o)
    return ok / len(pairs)


def run_benchmark(cfg: VoteConfig) -> list[VoteResult]:
    rng = np.random.RandomState(cfg.seed)
    results: list[VoteResult] = []

    token_pairs = _make_token_pairs(cfg.M_token, rng)

    def _token_items() -> ItemTable:
        items = ItemTable(cfg.D, seed=cfg.seed)
        for c, t in token_pairs:
            _ = items[c]
            _ = items[t]
        return items

    for vote in (False, True):
        label = "token_hash_vote" if vote else "token_hash_route"
        model = RPNColumn(
            cfg.D,
            _token_items(),
            cfg.n_neurons,
            mode="delta",
            vote_on_read=vote,
        )
        for c, t in token_pairs:
            model.learn(c, t)
        results.append(
            VoteResult(label, _token_accuracy(model, token_pairs), "token")
        )

    grid = _make_grid(cfg.n_colors, cfg.n_shapes)

    def _grid_items() -> ItemTable:
        items = ItemTable(cfg.D, seed=cfg.seed)
        _register_compositional(items, grid)
        return items

    for vote in (False, True):
        label = "color_vote" if vote else "color_route"
        model = ColorShardedColumn(
            cfg.D, _grid_items(), mode="delta", vote_on_read=vote
        )
        for color, shape, obj in grid:
            model.learn(color, shape, obj)
        results.append(VoteResult(label, _grid_accuracy(model, grid), "compositional"))

    for vote in (False, True):
        label = "hash_composite_vote" if vote else "hash_composite_route"
        model = HashShardedCompositeColumn(
            cfg.D,
            _grid_items(),
            cfg.n_neurons,
            mode="delta",
            vote_on_read=vote,
        )
        for color, shape, obj in grid:
            model.learn(color, shape, obj)
        results.append(
            VoteResult(label, _grid_accuracy(model, grid), "compositional")
        )

    return results


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[VoteResult], cfg: VoteConfig) -> Verdict:
    by = {r.name: r for r in results}
    token_route = by["token_hash_route"]
    token_vote = by["token_hash_vote"]
    color_route = by["color_route"]
    color_vote = by["color_vote"]
    hash_route = by["hash_composite_route"]
    hash_vote = by["hash_composite_vote"]

    checks: list[tuple[str, bool]] = [
        (
            f"token_hash_vote >= {cfg.token_vote_min_acc:.2f} ({token_vote.accuracy:.3f})",
            token_vote.accuracy >= cfg.token_vote_min_acc,
        ),
        (
            f"token vote >= route ({token_vote.accuracy:.3f} >= {token_route.accuracy:.3f})",
            token_vote.accuracy >= token_route.accuracy,
        ),
        (
            f"color_route >= {cfg.color_route_min_acc:.2f} ({color_route.accuracy:.3f})",
            color_route.accuracy >= cfg.color_route_min_acc,
        ),
        (
            f"color_vote <= {cfg.color_vote_max_acc:.2f} ({color_vote.accuracy:.3f})",
            color_vote.accuracy <= cfg.color_vote_max_acc,
        ),
        (
            f"color route beats vote ({color_route.accuracy:.3f} > {color_vote.accuracy:.3f})",
            color_route.accuracy > color_vote.accuracy,
        ),
        (
            f"hash_composite_route >= {cfg.hash_route_min_acc:.2f} ({hash_route.accuracy:.3f})",
            hash_route.accuracy >= cfg.hash_route_min_acc,
        ),
        (
            f"hash composite vote < route ({hash_vote.accuracy:.3f} < {hash_route.accuracy:.3f})",
            hash_vote.accuracy < hash_route.accuracy,
        ),
    ]

    lines: list[str] = []
    lines.append(
        f"Column vote benchmark (token M={cfg.M_token}, grid {cfg.n_colors}x{cfg.n_shapes}, "
        f"D={cfg.D}, N={cfg.n_neurons})"
    )
    lines.append("")
    lines.append(f"{'model':<24} {'scenario':<14} {'acc':>8}")
    for r in results:
        lines.append(f"{r.name:<24} {r.scenario:<14} {r.accuracy:>8.3f}")
    lines.append("")
    lines.append("Pass/fail gates:")
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  - Token hash vote ties route when each ctx is unique (M=256, 8 shards)."
    )
    lines.append(
        "  - Composite hash vote loses to route: wrong shards Born-pass on same ctx."
    )
    lines.append(
        "  - Color vote (shape ctx only) collapses — need semantic *where* routing."
    )
    lines.append(
        "  - B-3 lesson: vote replaces routing only when non-winners stay silent."
    )
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = VoteConfig()
    results = run_benchmark(cfg)
    verdict = evaluate_verdict(results, cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
