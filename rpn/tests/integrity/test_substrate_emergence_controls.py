"""Emergence gates for B-30 ("emergent roles", "neurogenesis under tension"). GEMINI.md section 5 (emergence).

An effect called "emergent" must be ABSENT in the fresh/untrained system and must depend on the task signal.
"""
from __future__ import annotations

import pytest
from controls import assert_gate, gate_absent_in_null
from maze_controls import DT, make_substrate

REFUTED = "B-30 REFUTED (docs/CLAIMS.md): "


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason=REFUTED + "a fresh untrained substrate is already classified 8/8 'Sensory Specialist' "
                                    "(initial impedance gives T=0.96 > the 0.85 classifier threshold)")
def test_roles_absent_before_training():
    roles = list(make_substrate(42).inspect_emergent_roles().values())
    frac = roles.count("Sensory Specialist") / len(roles)
    assert_gate(gate_absent_in_null(frac, 0.5, "fraction of 'Sensory Specialist' neurons in an UNTRAINED substrate"))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason=REFUTED + "the network grows to its neuron cap with zero input and zero reward (a timer, not tension)")
def test_neurogenesis_requires_task_signal():
    sub = make_substrate(42)
    n0 = len(sub.somas)
    for _ in range(1400):
        sub.step({i: 0j for i in range(8)}, reward=0.0, dt=DT)
    assert_gate(gate_absent_in_null(len(sub.somas) - n0, 1, "neurons grown with ZERO input and ZERO reward"))
