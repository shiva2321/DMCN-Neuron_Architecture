from neuron_v2.core.item_table import ItemTable
from neuron_v2.core.rpn import RPN

D = 8192


def test_store_and_recall():
    items = ItemTable(D, seed=7)
    for tok in ["key", "box", "truck", "garage"]:
        _ = items[tok]
    rpn = RPN(D, items, mode="delta")
    rpn.learn("key", "box")
    rpn.learn("box", "truck")
    assert rpn.read("key")[0] == "box"
    assert rpn.read("box")[0] == "truck"


def test_refuses_on_unseen_context():
    items = ItemTable(D, seed=7)
    for tok in ["key", "box", "truck", "garage"]:
        _ = items[tok]
    rpn = RPN(D, items, mode="delta")
    rpn.learn("key", "box")
    token, prob = rpn.read("garage")   # never written as a context
    assert token is None               # anti-hallucination: refuse rather than guess
