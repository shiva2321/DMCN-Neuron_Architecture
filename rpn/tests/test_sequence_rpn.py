"""Tests for ChainRPN multi-hop reads."""
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.sequence_rpn import ChainRPN, build_chain_edges, chain_endpoints


def test_chain_multihop_smoke():
    items = ItemTable(256, seed=4)
    edges = build_chain_edges(0, 4)
    for src, dst in edges:
        _ = items[src]
        _ = items[dst]
    chain = ChainRPN(256, items)
    for src, dst in edges:
        chain.learn_edge(src, dst)

    start, end1 = chain_endpoints(0, 4, 1)
    start, end2 = chain_endpoints(0, 4, 2)
    start, end3 = chain_endpoints(0, 4, 3)
    assert chain.read_hops(start, 1)[0] == end1
    assert chain.read_hops(start, 2)[0] == end2
    assert chain.read_hops(start, 3)[0] == end3
    assert chain.read_hops(start, 1)[0] != end2


def test_sequence_multihop_benchmark_smoke():
    from benchmarks.sequence_multihop import HopConfig, evaluate_verdict, run_benchmark

    cfg = HopConfig(D=256, n_chains=8, chain_length=4, min_hop3=0.75)
    verdict = evaluate_verdict(run_benchmark(cfg), cfg)
    assert verdict.passed


def test_multiband_benchmark_smoke():
    from benchmarks.multiband_denoise import BandConfig, run_benchmark

    cfg = BandConfig(D=256, M=24)
    assert run_benchmark(cfg).passed
