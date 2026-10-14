"""Tests for ResonantPredictiveColumn (full RPC)."""
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpc import ResonantPredictiveColumn

D = 256
PAIRS = [
    ("red", "circle", "apple"),
    ("red", "square", "box"),
    ("blue", "circle", "ball"),
    ("blue", "square", "block"),
]


def _make_rpc() -> ResonantPredictiveColumn:
    items = ItemTable(D, seed=9)
    for color, shape, obj in PAIRS:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]
    return ResonantPredictiveColumn(D, items)


def test_rpc_smoke_read_after_learn():
    rpc = _make_rpc()
    for color, shape, obj in PAIRS:
        rpc.learn(color, shape, obj)
    for color, shape, expected in PAIRS:
        token, prob = rpc.read(color, shape)
        assert token == expected
        assert prob > 0.0


def test_rpc_new_where_shard_isolated():
    items = ItemTable(D, seed=11)
    pairs = [(f"color_{i}", f"shape_{j}", f"obj_{i}_{j}") for i in range(3) for j in range(4)]
    for c, s, o in pairs:
        _ = items[c]
        _ = items[s]
        _ = items[o]

    rpc = ResonantPredictiveColumn(D, items)
    train = [p for p in pairs if p[0] != "color_2"]
    adapt = [p for p in pairs if p[0] == "color_2"]

    for c, s, o in train:
        rpc.learn(c, s, o)
    assert rpc.n_where_shards() == 2

    for c, s, o in adapt:
        rpc.learn(c, s, o)
    assert rpc.n_where_shards() == 3

    for c, s, o in train:
        assert rpc.read(c, s)[0] == o
    for c, s, o in adapt:
        assert rpc.read(c, s)[0] == o


def test_rpc_predictive_write_skip_on_relearn():
    rpc = _make_rpc()
    for color, shape, obj in PAIRS:
        rpc.learn(color, shape, obj)
    applied1, skipped1 = rpc.write_stats()
    assert applied1 > 0
    for color, shape, obj in PAIRS:
        rpc.learn(color, shape, obj)
    applied2, skipped2 = rpc.write_stats()
    assert skipped2 > skipped1


def test_rpc_plain_rpn_mode_still_works():
    items = ItemTable(D, seed=9)
    for color, shape, obj in PAIRS:
        _ = items[color]
        _ = items[shape]
        _ = items[obj]
    rpc = ResonantPredictiveColumn(D, items, memory="plain")
    for color, shape, obj in PAIRS:
        rpc.learn(color, shape, obj)
    for color, shape, expected in PAIRS:
        assert rpc.read(color, shape)[0] == expected
