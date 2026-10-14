"""Reference frames — bind fillers into named slots (what / where) for composition."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..substrate.phasor import bind, bundle, permute, random_hv
from .item_table import ItemTable


def _role_seed(name: str) -> int:
    digest = hashlib.md5(f"compositional-role:{name}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


@dataclass(frozen=True)
class SlotSpec:
    """Named slot with a cyclic permute offset on its role HV."""

    name: str
    permute_offset: int
    role_key: str


class ReferenceFrame:
    """Binds token fillers into role-tagged slots and composes a query context."""

    SLOT_WHERE = "where"
    SLOT_WHAT = "what"

    def __init__(self, D: int, slots: list[SlotSpec]):
        if not slots:
            raise ValueError("slots must be non-empty")
        self.D = D
        self.slots = tuple(slots)
        self._roles: dict[str, np.ndarray] = {
            spec.name: random_hv(D, seed=_role_seed(spec.role_key))
            for spec in self.slots
        }

    @classmethod
    def color_shape(cls, D: int) -> ReferenceFrame:
        """Default frame: color = where, shape = what (matches B-1 role seeds)."""
        return cls(
            D,
            [
                SlotSpec(cls.SLOT_WHERE, 1, "role_color"),
                SlotSpec(cls.SLOT_WHAT, 2, "role_shape"),
            ],
        )

    @classmethod
    def agent_action(cls, D: int) -> ReferenceFrame:
        """Alternate frame: agent = where, action = what (distinct role seeds)."""
        return cls(
            D,
            [
                SlotSpec(cls.SLOT_WHERE, 3, "role_agent"),
                SlotSpec(cls.SLOT_WHAT, 4, "role_action"),
            ],
        )

    def bind_slot(self, items: ItemTable, slot: str, token: str) -> np.ndarray:
        spec = next(s for s in self.slots if s.name == slot)
        role = permute(self._roles[slot], spec.permute_offset)
        return bind(role, items[token])

    def compose(self, items: ItemTable, fillers: dict[str, str]) -> np.ndarray:
        parts = [self.bind_slot(items, spec.name, fillers[spec.name]) for spec in self.slots]
        return bundle(*parts)
