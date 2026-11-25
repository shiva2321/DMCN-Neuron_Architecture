"""Null-control gates for the maze claims B-30 / B-31 (docs/CLAIMS.md).  GEMINI.md section 5.

The claim under test (B-31): "a morphogenetic CRFN trained on 10 demonstrations navigates the maze autonomously,
with no teacher and no cheats". Each test runs the system and the dumbest alternative that could produce the
same number, under the frozen reference protocol in ``maze_controls.py``.

Tests marked ``xfail(strict=True)`` document a REFUTED claim. If one starts to pass, the suite fails (XPASS strict):
update docs/CLAIMS.md and remove the marker in the same change - never silently.
"""
from __future__ import annotations

import pytest
from controls import (assert_gate, gate_beats_controls, gate_depends_on_inputs, rate)
from maze_controls import (CONSTANT_DIRECTIONS, CORRIDOR_WAYPOINTS, N_DEMOS, beacon_policy, build_demonstration_path,
                           constant_policy, make_substrate, maze1, network_policy, open_arena, snapshot,
                           success_outcomes, train_with_demonstrations)

N_EVAL = 6
REFUTED_B31 = "B-31 REFUTED (docs/CLAIMS.md): the claim does not survive this control"


@pytest.fixture(scope="module")
def trained():
    sub = make_substrate(42)
    snap = train_with_demonstrations(sub, maze1(), CORRIDOR_WAYPOINTS, build_demonstration_path(), N_DEMOS, seed=7)
    return sub, snap


@pytest.fixture(scope="module")
def untrained():
    sub = make_substrate(42)
    return sub, snapshot(sub)


def net_rate(sub_snap, maze, physics="slide", blind=False, n=N_EVAL):
    sub, snap = sub_snap
    return rate(success_outcomes(lambda s: network_policy(sub, snap, s, blind=blind), maze, n, physics))


def test_frozen_protocol_headline_reproduces_as_measured(trained):
    """Honest reading of the headline: under the final (sliding) protocol the trained net does reach the goal."""
    r = net_rate(trained, maze1())
    assert r.p >= 0.8, f"headline no longer reproduces: {r}"


def test_trained_beats_untrained(trained, untrained):
    """A real (but not sufficient) effect: training changes behaviour relative to the untrained substrate."""
    assert_gate(gate_beats_controls(net_rate(trained, maze1()), {"untrained": net_rate(untrained, maze1())}, 0.3))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REFUTED_B31 + " - no-network policies (constant push / goal-beacon) also solve the maze")
def test_system_beats_every_no_network_control(trained):
    maze = maze1()
    controls = {f"const-{k}": rate(success_outcomes(lambda s, f=f: constant_policy(f), maze, N_EVAL))
                for k, f in CONSTANT_DIRECTIONS.items()}
    controls["goal-beacon-only"] = rate(success_outcomes(lambda s: beacon_policy(), maze, N_EVAL))
    assert_gate(gate_beats_controls(net_rate(trained, maze), controls, min_margin=0.2))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REFUTED_B31 + " - the trained net still succeeds with every sensor zeroed")
def test_system_needs_its_sensors(trained):
    assert_gate(gate_depends_on_inputs(net_rate(trained, maze1()), net_rate(trained, maze1(), blind=True)))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REFUTED_B31 + " - result collapses under the original (pre-684fb18) physics")
def test_survives_original_physics(trained):
    r = net_rate(trained, maze1(), physics="bounce")
    assert_gate([] if r.p >= 0.8 else [f"success under original physics only {r}"])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REFUTED_B31 + " - loses to a 4-line goal-beacon controller in an open arena")
def test_beats_goal_beacon_heuristic_in_open_arena(trained):
    net_hits, beacon_hits = [], []
    for goal in [(3.5, 3.5), (0.5, 3.5), (0.5, 0.5), (3.5, 0.5)]:
        arena = open_arena(goal)
        net_hits += success_outcomes(lambda s: network_policy(trained[0], trained[1], s), arena, 4)
        beacon_hits += success_outcomes(lambda s: beacon_policy(), arena, 4)
    assert_gate(gate_beats_controls(rate(net_hits), {"goal-beacon-only": rate(beacon_hits)}, min_margin=0.0))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason=REFUTED_B31 + " - training on a time-shuffled (decoupled) teacher succeeds as often as the real one")
def test_demonstrations_carry_information(trained):
    """Decoupled-teacher control (GEMINI.md section 5): identical demonstrations with the teacher's commands shuffled in time.

    If the shuffled teacher works as well as the real one, the demonstrations contribute nothing task-specific and the
    maze is being solved by the environment/physics, not by what was demonstrated.
    """
    sub = make_substrate(42)
    snap = train_with_demonstrations(sub, maze1(), CORRIDOR_WAYPOINTS, build_demonstration_path(), N_DEMOS,
                                     seed=7, shuffle_teacher=True)
    assert_gate(gate_beats_controls(net_rate(trained, maze1()),
                                    {"shuffled-teacher": net_rate((sub, snap), maze1())}, min_margin=0.3))
