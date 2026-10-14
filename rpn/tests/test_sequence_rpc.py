"""Tests for SequenceRPC and B-11 composition."""
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpc import ResonantPredictiveColumn
from neuron_v2.core.sequence_rpc import SequenceRPC


def test_sequence_rpc_two_hop_smoke():
    items = ItemTable(256, seed=5)
    pairs = [
        ("red", "circle", "apple", "eat"),
        ("blue", "square", "block", "push"),
    ]
    for c, s, o, a in pairs:
        _ = items[c]
        _ = items[s]
        _ = items[o]
        _ = items[a]

    stack = SequenceRPC(256, items)
    for c, s, o, a in pairs:
        stack.learn(c, s, o, a)

    for c, s, o, a in pairs:
        assert stack.read_object(c, s)[0] == o
        assert stack.read_action(c, s)[0] == a
    assert stack.chain.read_hops("red", 1)[0] != "eat"


def test_rpc_multiband_memory_smoke():
    items = ItemTable(256, seed=6)
    pairs = [("red", "circle", "apple"), ("blue", "square", "block")]
    for c, s, o in pairs:
        _ = items[c]
        _ = items[s]
        _ = items[o]

    rpc = ResonantPredictiveColumn(256, items, memory="multiband", multiband_K=3)
    for c, s, o in pairs:
        rpc.learn(c, s, o)
    for c, s, o in pairs:
        assert rpc.read(c, s)[0] == o


def test_sequence_composition_benchmark_smoke():
    from benchmarks.sequence_composition import SeqCompConfig, evaluate_verdict, run_benchmark

    cfg = SeqCompConfig(D=256, n_colors=4, n_shapes=4)
    assert evaluate_verdict(run_benchmark(cfg), cfg).passed


def test_rpc_multiband_benchmark_smoke():
    from benchmarks.rpc_multiband import RpcMbConfig, run_benchmark

    cfg = RpcMbConfig(D=256, n_colors=4, n_shapes=4)
    assert run_benchmark(cfg).passed


def test_rpc_kuramoto_memory_smoke():
    items = ItemTable(256, seed=7)
    pairs = [("red", "circle", "apple"), ("blue", "square", "block")]
    for c, s, o in pairs:
        _ = items[c]
        _ = items[s]
        _ = items[o]

    rpc = ResonantPredictiveColumn(256, items, memory="kuramoto", k=4, kuramoto_K=3)
    for c, s, o in pairs:
        rpc.learn(c, s, o)
    for c, s, o in pairs:
        assert rpc.read(c, s)[0] == o


def test_rpc_kuramoto_benchmark_smoke():
    from benchmarks.rpc_kuramoto import RpcKmConfig, run_benchmark

    cfg = RpcKmConfig(
        D=256,
        n_colors=4,
        n_shapes=4,
        min_predictive_acc=0.85,
        max_acc_gap=0.10,
    )
    assert run_benchmark(cfg).passed
