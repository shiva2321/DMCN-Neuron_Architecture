"""Sequence multi-hop benchmark (B-10): chained associative reads.

Run: python -m benchmarks.sequence_multihop
"""
from __future__ import annotations

from dataclasses import dataclass

from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.sequence_rpn import ChainRPN, build_chain_edges, chain_endpoints


@dataclass
class HopConfig:
    D: int = 512
    seed: int = 42
    n_chains: int = 48
    chain_length: int = 5
    min_hop1: float = 0.98
    min_hop2: float = 0.95
    min_hop3: float = 0.90
    max_one_step_on_hop2: float = 0.15


@dataclass
class HopResult:
    hops: int
    accuracy: float
    one_step_baseline: float | None = None


def _register_chains(items: ItemTable, cfg: HopConfig) -> list[list[tuple[str, str]]]:
    chains: list[list[tuple[str, str]]] = []
    for cid in range(cfg.n_chains):
        edges = build_chain_edges(cid, cfg.chain_length)
        for src, dst in edges:
            _ = items[src]
            _ = items[dst]
        chains.append(edges)
    return chains


def _eval_hops(chain: ChainRPN, cfg: HopConfig, hops: int) -> HopResult:
    ok = 0
    one_ok = 0
    for cid in range(cfg.n_chains):
        start, expected = chain_endpoints(cid, cfg.chain_length, hops)
        got, _ = chain.read_hops(start, hops)
        if got == expected:
            ok += 1
        if hops >= 2:
            one, _ = chain.read_hops(start, 1)
            if one == expected:
                one_ok += 1
    acc = ok / cfg.n_chains
    baseline = (one_ok / cfg.n_chains) if hops >= 2 else None
    return HopResult(hops=hops, accuracy=acc, one_step_baseline=baseline)


def run_benchmark(cfg: HopConfig) -> list[HopResult]:
    items = ItemTable(cfg.D, seed=cfg.seed)
    chains = _register_chains(items, cfg)
    model = ChainRPN(cfg.D, items)
    for edges in chains:
        for src, dst in edges:
            model.learn_edge(src, dst)

    return [
        _eval_hops(model, cfg, 1),
        _eval_hops(model, cfg, 2),
        _eval_hops(model, cfg, 3),
    ]


@dataclass
class Verdict:
    passed: bool
    lines: list[str]


def evaluate_verdict(results: list[HopResult], cfg: HopConfig) -> Verdict:
    h1, h2, h3 = results
    checks = [
        (f"1-hop >= {cfg.min_hop1:.2f} ({h1.accuracy:.3f})", h1.accuracy >= cfg.min_hop1),
        (f"2-hop >= {cfg.min_hop2:.2f} ({h2.accuracy:.3f})", h2.accuracy >= cfg.min_hop2),
        (f"3-hop >= {cfg.min_hop3:.2f} ({h3.accuracy:.3f})", h3.accuracy >= cfg.min_hop3),
    ]
    if h2.one_step_baseline is not None:
        checks.append(
            (
                f"1-step on 2-hop queries <= {cfg.max_one_step_on_hop2:.2f} "
                f"({h2.one_step_baseline:.3f})",
                h2.one_step_baseline <= cfg.max_one_step_on_hop2,
            )
        )
    lines = [
        f"Sequence multi-hop (chains={cfg.n_chains}, length={cfg.chain_length}, D={cfg.D})",
        "",
        f"{'hops':>6} {'accuracy':>10}",
    ]
    for r in results:
        lines.append(f"{r.hops:>6} {r.accuracy:>10.3f}")
    if h2.one_step_baseline is not None:
        lines.append(f"{'1-step@2-hop':>6} {h2.one_step_baseline:>10.3f}")
    lines.append("")
    lines.append("Pass/fail gates:")
    for msg, ok in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    passed = all(ok for _, ok in checks)
    lines.append("")
    lines.append("OVERALL: PASS" if passed else "OVERALL: FAIL")
    return Verdict(passed=passed, lines=lines)


def main() -> int:
    cfg = HopConfig()
    verdict = evaluate_verdict(run_benchmark(cfg), cfg)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
