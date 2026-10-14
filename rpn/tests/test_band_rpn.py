"""Tests for MultiBandRPN."""
import numpy as np

from neuron_v2.core.band_rpn import MultiBandRPN, band_slice, project_band, random_band_scramble
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpn import RPN


def test_project_band_is_disjoint():
    v = np.exp(1j * np.linspace(0, 1, 12)).astype(np.complex64)
    p0 = project_band(v, 0, 3)
    p1 = project_band(v, 1, 3)
    assert np.linalg.norm(p0[band_slice(12, 1, 3)]) < 1e-6
    assert np.linalg.norm(p1[band_slice(12, 0, 3)]) < 1e-6


def test_multiband_learn_read_smoke():
    items = ItemTable(256, seed=1)
    pairs = [(f"c{i}", f"t{i}") for i in range(8)]
    for c, t in pairs:
        _ = items[c]
        _ = items[t]
    mb = MultiBandRPN(256, items, K=3)
    for c, t in pairs:
        mb.learn(c, t)
    for c, t in pairs:
        assert mb.read(c)[0] == t


def test_sharded_multiband_routes_per_band():
    items = ItemTable(256, seed=2)
    mb = MultiBandRPN(256, items, K=3, redundant=False)
    for band in range(3):
        ctx, tgt = f"band{band}_ctx", f"band{band}_tgt"
        _ = items[ctx]
        _ = items[tgt]
        mb.learn(ctx, tgt, band=band)
        assert mb.read(ctx)[0] == tgt


def test_redundant_multiband_partial_band_read():
    items = ItemTable(256, seed=3)
    pairs = [(f"c{i}", f"t{i}") for i in range(8)]
    for c, t in pairs:
        _ = items[c]
        _ = items[t]
    mb = MultiBandRPN(256, items, K=3, redundant=True)
    for c, t in pairs:
        mb.learn(c, t)
    rng = np.random.RandomState(1)
    for c, t in pairs:
        ctx, clean = random_band_scramble(items[c], 3, 0.6, rng)
        if clean:
            assert mb.read(c, ctx_hv=ctx, active_bands=clean)[0] == t
