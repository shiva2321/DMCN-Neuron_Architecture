"""Empirical Benchmark: Continuous Sensorimotor Tracking via Unified Resonant Learning.

Evaluates the Continuous Resonant Field Neuron (CRFN) unified learning algorithm:
- Unsupervised Predictive Coding: Destructive wave cancellation of sensory prediction errors.
- Three-Factor Reinforcement Learning: Neuromodulated dopamine RPE acting on resonant eligibility traces.
- Autonomous Structural Plasticity: Spontaneous neurogenesis and activity-dependent pruning.

Zero backpropagation. Zero external vector codebooks. Zero matrix factorizations.
Pure physical wave transmission, impedance matching, and Stuart-Landau limit cycles.
"""
from __future__ import annotations

import json
import time
import numpy as np
from pathlib import Path

from neuron_v2.crfn.crfn_rl_predictive import NeuromodulatedCRFNNetwork


def run_sensorimotor_tracking_benchmark(
    n_epochs: int = 15,
    steps_per_epoch: int = 200,
    dt: float = 0.02,
    seed: int = 42,
) -> dict:
    """Benchmark the unified CRFN network on a continuous 1D physical tracking task."""
    rng = np.random.RandomState(seed)
    omega_target = 1.50  # Target oscillation frequency (rad/s)

    # 1. Instantiate the unified CRFN network
    net = NeuromodulatedCRFNNetwork(
        n_sensory=1,
        n_recurrent=3,
        n_motor=1,
        base_omega=omega_target,
        gamma_discount=0.90,
        lr_rl=1.20,
        lr_pred=0.08,
        theta_grow=0.80,
        theta_prune=0.15,
        exploration_std=0.08,
        seed=seed,
    )

    epoch_rewards = []
    epoch_tracking_errors = []
    epoch_prediction_errors = []
    epoch_motor_transmissions = []
    epoch_neuron_counts = []
    epoch_link_counts = []

    print("=" * 78)
    print("CRFN UNIFIED SENSORIMOTOR TRACKING BENCHMARK (PREDICTIVE CODING + RL)")
    print("=" * 78)
    print(f"{'Epoch':<7} | {'Cum Reward':<12} | {'Tracking RMSE':<14} | {'Pred Error':<12} | {'Motor T':<9} | {'Neurons':<8} | {'Links':<7}")
    print("-" * 78)

    t0_wall = time.perf_counter()

    for epoch in range(n_epochs):
        net.reset()
        # Physical agent state: 1D mass with damping: x_dot_dot + beta * x_dot = u(t)
        x_agent = 0.0
        v_agent = 0.0
        beta_damping = 0.50
        mass = 1.0

        cum_reward = 0.0
        tracking_errors = []
        pred_errors = []

        for step in range(steps_per_epoch):
            t = step * dt
            # Oscillating target trajectory: y(t) = sin(omega * t)
            y_target = float(np.sin(omega_target * t))
            v_target = float(omega_target * np.cos(omega_target * t))

            # Sensory input is the tracking discrepancy wave: (y_target - x_agent) modulated by carrier
            discrepancy = y_target - x_agent
            sensory_wave = discrepancy * np.exp(1j * omega_target * t)

            # Continuous reward based on spatial tracking proximity
            instant_error = abs(discrepancy)
            reward = float(np.exp(-instant_error ** 2 / 0.5) - 0.20)
            cum_reward += reward
            tracking_errors.append(instant_error)

            # Step the CRFN network
            out = net.step({0: sensory_wave}, reward=reward, dt=dt)
            pred_errors.append(float(out["prediction_error"]))

            # Motor control wave applies physical force to the agent
            u_motor = float(out["motor_action"])

            # Numerical integration of agent mechanics
            a_agent = (u_motor * 3.0 - beta_damping * v_agent) / mass
            v_agent += a_agent * dt
            x_agent += v_agent * dt

        # Compute epoch statistics
        rmse = float(np.sqrt(np.mean(np.square(tracking_errors))))
        mean_pred_err = float(np.mean(pred_errors))

        # Average power transmission across motor synapses
        motor_links = [l for pair, l in net.links.items() if pair[1] in net.motor_ids]
        mean_motor_T = float(np.mean([l.power_transmission for l in motor_links])) if motor_links else 0.0

        epoch_rewards.append(cum_reward)
        epoch_tracking_errors.append(rmse)
        epoch_prediction_errors.append(mean_pred_err)
        epoch_motor_transmissions.append(mean_motor_T)
        epoch_neuron_counts.append(len(net.somas))
        epoch_link_counts.append(len(net.links))

        print(
            f"{epoch + 1:<7} | {cum_reward:<12.2f} | {rmse:<14.4f} | {mean_pred_err:<12.4f} | "
            f"{mean_motor_T:<9.3f} | {len(net.somas):<8} | {len(net.links):<7}"
        )

    t_elapsed = time.perf_counter() - t0_wall
    print("-" * 78)
    print(f"Benchmark completed in {t_elapsed:.2f} seconds.")
    print("=" * 78)

    # Calculate key benchmark outcomes
    reward_gain = epoch_rewards[-1] - epoch_rewards[0]
    rmse_reduction = (epoch_tracking_errors[0] - epoch_tracking_errors[-1]) / (epoch_tracking_errors[0] + 1e-8)
    motor_T_gain = epoch_motor_transmissions[-1] - epoch_motor_transmissions[0]

    results = {
        "n_epochs": n_epochs,
        "steps_per_epoch": steps_per_epoch,
        "dt": dt,
        "elapsed_seconds": t_elapsed,
        "initial_reward": float(epoch_rewards[0]),
        "final_reward": float(epoch_rewards[-1]),
        "reward_gain": float(reward_gain),
        "initial_tracking_rmse": float(epoch_tracking_errors[0]),
        "final_tracking_rmse": float(epoch_tracking_errors[-1]),
        "tracking_rmse_reduction_pct": float(rmse_reduction * 100.0),
        "initial_motor_transmission": float(epoch_motor_transmissions[0]),
        "final_motor_transmission": float(epoch_motor_transmissions[-1]),
        "motor_transmission_gain": float(motor_T_gain),
        "initial_neuron_count": int(epoch_neuron_counts[0]),
        "final_neuron_count": int(epoch_neuron_counts[-1]),
        "initial_link_count": int(epoch_link_counts[0]),
        "final_link_count": int(epoch_link_counts[-1]),
        "epoch_rewards": [float(r) for r in epoch_rewards],
        "epoch_tracking_errors": [float(e) for e in epoch_tracking_errors],
        "epoch_prediction_errors": [float(p) for p in epoch_prediction_errors],
        "epoch_motor_transmissions": [float(t) for t in epoch_motor_transmissions],
    }

    # Save to disk
    out_path = Path("crfn_learning_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results serialized to {out_path.resolve()}")

    return results


if __name__ == "__main__":
    run_sensorimotor_tracking_benchmark()
