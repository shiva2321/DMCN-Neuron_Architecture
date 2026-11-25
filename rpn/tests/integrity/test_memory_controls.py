"""Capacity-claim control for B-17 (RCC "laminar column"). GEMINI.md section 5 (memory/retrieval), section 7."""
from __future__ import annotations

import numpy as np
import pytest
from controls import Rate, assert_gate, gate_beats_controls
from neuron_v2.core.delta_memory import DeltaMemory
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.laminar_column import LaminarCorticalColumn
from neuron_v2.substrate.phasor import bind, normalise, permute, sim_batch
from benchmarks.rcc_laminar_generalization import generate_grid_dataset, register_items


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B-17 REFUTED (docs/CLAIMS.md): one unsharded DeltaMemory with the column's own bound key matches "
                          "the 'laminar column' (the 28% monolith baseline used a weaker bundled key)")
def test_rcc_not_matched_by_plain_memory():
    D = 256
    pairs = generate_grid_dataset(8, 16)
    items = ItemTable(D, seed=42)
    register_items(items, pairs)
    rcc = LaminarCorticalColumn(D, items, n_shards=8, seed=42)
    for w, t, o in pairs:
        rcc.learn(w, t, o)
    rcc_hits = sum(rcc.read(w, t)[0] == o for w, t, o in pairs)

    frame = rcc.frame
    mem = DeltaMemory(D=D, mode="delta", lr=1.0)             # ONE memory: no shards, no RCPN units, no layers

    def key(w, t):
        return bind(bind(permute(frame.role_where, 1), items[w]), bind(permute(frame.role_what, 2), items[t]))

    for w, t, o in pairs:
        mem.write(key(w, t), items[o])
    tokens, mat = items.matrix()

    def read(w, t):
        s = sim_batch(mat, normalise(mem.predict(key(w, t))))
        i = int(np.argmax(s))
        return tokens[i] if s[i] >= 0.10 else None

    plain_hits = sum(read(w, t) == o for w, t, o in pairs)
    assert_gate(gate_beats_controls(Rate(rcc_hits, len(pairs)),
                                    {"plain unsharded memory, same key": Rate(plain_hits, len(pairs))}, min_margin=0.2))
