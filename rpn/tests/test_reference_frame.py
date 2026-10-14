"""Tests for ReferenceFrame composition."""
from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.reference_frame import ReferenceFrame
from benchmarks.baselines import RoleBoundCompositeCtx
from neuron_v2.substrate.phasor import sim


def test_agent_action_frame_compose_differs_from_color_shape():
    items = ItemTable(128, seed=5)
    _ = items["agent_0"]
    _ = items["action_0"]
    _ = items["color_0"]
    _ = items["shape_0"]
    aa = ReferenceFrame.agent_action(128)
    cs = ReferenceFrame.color_shape(128)
    a = aa.compose(items, {aa.SLOT_WHERE: "agent_0", aa.SLOT_WHAT: "action_0"})
    b = cs.compose(items, {cs.SLOT_WHERE: "color_0", cs.SLOT_WHAT: "shape_0"})
    assert sim(a, b) < 0.5


def test_reference_frame_matches_role_bound_composite():
    D = 256
    items = ItemTable(D, seed=3)
    _ = items["red"]
    _ = items["circle"]

    frame = ReferenceFrame.color_shape(D)
    legacy = RoleBoundCompositeCtx(D, items)

    fillers = {ReferenceFrame.SLOT_WHERE: "red", ReferenceFrame.SLOT_WHAT: "circle"}
    a = frame.compose(items, fillers)
    b = legacy("red", "circle")
    assert sim(a, b) > 0.999
