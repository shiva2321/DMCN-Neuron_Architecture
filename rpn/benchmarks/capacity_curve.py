"""Headline figure: delta vs additive recall + surface RMS as repeated writes pile up."""
import numpy as np

from neuron_v2.substrate.phasor import random_hv, sim
from neuron_v2.core.delta_memory import DeltaMemory

# D=512 keeps the headline table runnable on CPU (~seconds). Tests cover higher D.
D, M = 512, 32


def run(mode: str, epochs: int) -> tuple[float, float]:
    rng = np.random.RandomState(42)
    ctxs = [random_hv(D, seed=int(rng.randint(1, 2**31 - 1))) for _ in range(M)]
    tgts = [random_hv(D, seed=int(rng.randint(1, 2**31 - 1))) for _ in range(M)]
    mem = DeltaMemory(D, mode=mode)
    for _ in range(epochs):
        for c, t in zip(ctxs, tgts):
            mem.write(c, t)
    recall = float(np.mean([sim(mem.recall(c), t) for c, t in zip(ctxs, tgts)]))
    return recall, mem.rms()


if __name__ == "__main__":
    print(f"{'epochs':>6} | {'delta_recall':>12} {'delta_rms':>10} | {'add_recall':>10} {'add_rms':>10}")
    for e in [1, 2, 4, 8, 16, 32]:
        dr, drms = run("delta", e)
        ar, arms = run("additive", e)
        print(f"{e:>6} | {dr:>12.3f} {drms:>10.3f} | {ar:>10.3f} {arms:>10.3f}")
