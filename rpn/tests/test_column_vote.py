"""Smoke tests for B-3 column vote read policy."""

from neuron_v2.core.column import ColorShardedColumn, HashShardedCompositeColumn, vote_read_ctx
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpn import RPN
from benchmarks.baselines import RPNColumn


def test_vote_read_ctx_picks_highest_prob():
    D = 128
    items = ItemTable(D, seed=1)
    for tok in ("a", "b", "c"):
        _ = items[tok]
    neurons = [RPN(D, items, mode="delta") for _ in range(3)]
    ctx = items["a"]
    neurons[1].mem.write(ctx, items["b"])
    token, prob = vote_read_ctx(neurons, D, items, ctx)
    assert token == "b"
    assert prob > 0.0


def test_rpn_column_vote_matches_route_small_M():
    D = 256
    items = ItemTable(D, seed=3)
    pairs = [(f"ctx_{i}", f"tgt_{i}") for i in range(16)]
    for c, t in pairs:
        _ = items[c]
        _ = items[t]

    routed = RPNColumn(D, items, n_neurons=4, mode="delta", vote_on_read=False)
    voting = RPNColumn(D, items, n_neurons=4, mode="delta", vote_on_read=True)
    for c, t in pairs:
        routed.learn(c, t)
        voting.learn(c, t)

    for c, _ in pairs:
        assert routed.read(c) == voting.read(c)


def test_color_vote_fails_without_color_routing():
    D = 256
    pairs = [
        ("red", "circle", "apple"),
        ("blue", "circle", "ball"),
    ]
    items = ItemTable(D, seed=7)
    for color, shape, obj in pairs:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]

    routed = ColorShardedColumn(D, items, vote_on_read=False)
    voting = ColorShardedColumn(D, items, vote_on_read=True)
    for color, shape, obj in pairs:
        routed.learn(color, shape, obj)
        voting.learn(color, shape, obj)

    assert routed.read("red", "circle")[0] == "apple"
    assert routed.read("blue", "circle")[0] == "ball"
    # Both shards share shape ctx "circle" — vote can pick wrong shard.
    assert voting.read("red", "circle")[0] != "apple" or voting.read("blue", "circle")[0] != "ball"


def test_hash_composite_route_beats_vote_under_interference():
    D = 256
    pairs = [
        (f"color_{i}", f"shape_{j}", f"obj_{i}_{j}")
        for i in range(4)
        for j in range(4)
    ]
    items = ItemTable(D, seed=11)
    for color, shape, obj in pairs:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]

    routed = HashShardedCompositeColumn(D, items, 4, vote_on_read=False)
    voting = HashShardedCompositeColumn(D, items, 4, vote_on_read=True)
    for color, shape, obj in pairs:
        routed.learn(color, shape, obj)
        voting.learn(color, shape, obj)

    routed_ok = sum(1 for c, s, o in pairs if routed.read(c, s)[0] == o)
    vote_ok = sum(1 for c, s, o in pairs if voting.read(c, s)[0] == o)
    assert routed_ok == len(pairs)
    assert vote_ok < routed_ok
