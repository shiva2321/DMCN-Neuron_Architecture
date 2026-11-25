"""Unit, causality, numerical parity, homeostasis, and mechanism tests for DMCN."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neuron_v2.dmcn import (
    AblationConfig,
    DMCNConfig,
    DMCNLayer,
    DMCNNetwork,
    DMCNTorchLayer,
    DMCNTorchNetwork,
    FactorizedBilinearDMCNLayer,
    PDNCActorCritic,
    PDNCConfig,
    PDNCLayer,
    PDNCNetwork,
    PDNCTorchNetwork,
    SelfGatingConfig,
    SelfGatingDMCNLayer,
    SwarmBoidNetwork,
    SwarmConfig,
    UnsupOjaDendriticLayer,
    count_layer_ops,
)


def test_strict_online_causality():
    """Altering future inputs x[t+1:] must produce bit-identical outputs at all steps <= t."""
    cfg = DMCNConfig(d_in=4, d_out=3, n_branches=3, n_spine_scales=3, use_nonmonotonic=True, seed=11)
    rng = np.random.RandomState(11)
    seq_a = rng.randn(20, 4)
    seq_b = seq_a.copy()
    seq_b[10:] = rng.randn(10, 4) * 5.0

    layer_a = DMCNLayer(cfg)
    layer_b = DMCNLayer(cfg)

    for t in range(10):
        ya = layer_a.step(seq_a[t], dt=0.5)
        yb = layer_b.step(seq_b[t], dt=0.5)
        np.testing.assert_allclose(ya, yb, atol=0.0, rtol=0.0)


def test_numpy_pytorch_numerical_parity():
    """DMCNTorchLayer (CPU & CUDA if available) must match NumPy DMCNLayer on forward + local learn."""
    cfg = DMCNConfig(
        d_in=6,
        d_out=4,
        n_branches=3,
        n_spine_scales=2,
        use_nonmonotonic=True,
        soma_activation="tanh",
        seed=23,
    )
    np_layer = DMCNLayer(cfg)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    rng = np.random.RandomState(23)
    xs = rng.randn(15, 6)
    ts = rng.randn(15, 4)

    for dev in devices:
        np_copy = DMCNLayer(cfg)
        pt_layer = DMCNTorchLayer(cfg, device=dev, dtype=torch.float64)
        pt_layer.load_from_numpy(np_copy)

        for step_idx in range(15):
            y_np = np_copy.step(xs[step_idx], dt=1.0)
            y_pt = pt_layer(torch.tensor(xs[step_idx], dtype=torch.float64, device=dev), dt=1.0).cpu().numpy()
            np.testing.assert_allclose(y_pt, y_np, atol=1e-6, rtol=1e-5)

            err_np = ts[step_idx] - y_np
            err_pt = torch.tensor(err_np, dtype=torch.float64, device=dev)
            fb_np = np_copy.learn_local(err_np, mod=1.0)
            fb_pt = pt_layer.learn_local(err_pt, mod=1.0).cpu().numpy()
            np.testing.assert_allclose(fb_pt, fb_np, atol=1e-6, rtol=1e-5)

        np.testing.assert_allclose(pt_layer.W_fast.cpu().numpy(), np_copy.W_fast, atol=1e-6)
        np.testing.assert_allclose(pt_layer.W_slow.cpu().numpy(), np_copy.W_slow, atol=1e-6)
        np.testing.assert_allclose(pt_layer.B_fb.cpu().numpy(), np_copy.B_fb, atol=1e-6)

    assert np_layer.d_psi == 6 * (1 + 2)


def test_dormant_branch_recruitment_absent_on_zero_and_noise():
    """Null control: zero input and stationary white noise must trigger 0 false branch recruitments."""
    cfg = DMCNConfig(
        d_in=6,
        d_out=3,
        n_branches=4,
        n_spine_scales=0,
        theta_novel=-2.0,
        theta_err=0.5,
        theta_coh=0.40,
        recruit_sustained=5,
        seed=7,
    )
    # 1. Zero input + zero error -> zero recruitments
    layer_zero = DMCNLayer(cfg)
    for _ in range(150):
        layer_zero.step(np.zeros(6))
        layer_zero.learn_local(np.zeros(3))
    assert int(np.sum(layer_zero.recruited)) == 1
    assert len(layer_zero.recruit_events) == 0

    # 2. Stationary white noise errors -> incoherent errors must not trigger coherent recruitment
    rng = np.random.RandomState(7)
    layer_noise = DMCNLayer(cfg)
    for _ in range(200):
        layer_noise.step(rng.randn(6))
        layer_noise.learn_local(rng.randn(3))
    assert int(np.sum(layer_noise.recruited)) == 1


def test_homeostasis_bounds_gain_shock():
    """Under a sudden 100x input gain shock, homeostasis keeps internal dendritic drive bounded."""
    cfg_intact = DMCNConfig(d_in=4, d_out=2, n_branches=1, n_spine_scales=2, soma_activation="linear", seed=5)
    cfg_ablated = DMCNConfig(
        d_in=4,
        d_out=2,
        n_branches=1,
        n_spine_scales=2,
        soma_activation="linear",
        seed=5,
        ablation=AblationConfig(ablate_homeostasis=True),
    )
    l_intact = DMCNLayer(cfg_intact)
    l_ablated = DMCNLayer(cfg_ablated)

    rng = np.random.RandomState(5)
    for _ in range(60):
        x = rng.randn(4)
        l_intact.step(x)
        l_ablated.step(x)

    # Apply 100x gain shock
    intact_norms = []
    ablated_norms = []
    for _ in range(80):
        x_shock = rng.randn(4) * 100.0
        intact_norms.append(float(np.linalg.norm(l_intact.step(x_shock))))
        ablated_norms.append(float(np.linalg.norm(l_ablated.step(x_shock))))

    assert np.mean(intact_norms[-20:]) < 0.2 * np.mean(ablated_norms[-20:])


def test_kolen_pollack_local_feedback_alignment():
    """Activity-gated Kolen-Pollack must increase cosine alignment between B_fb and W_eff^T."""
    cfg = DMCNConfig(
        d_in=8,
        d_out=4,
        n_branches=1,
        n_spine_scales=0,
        soma_activation="linear",
        eta_fast=0.05,
        eta_fb=0.05,
        lambda_fb=0.005,
        seed=99,
    )
    layer = DMCNLayer(cfg)
    rng = np.random.RandomState(99)
    w_true = rng.randn(4, 8) * 0.5

    def align_cos() -> float:
        w_eff = (layer.W_slow[0] + layer.W_fast[0]).T
        b_fb = layer.B_fb
        return float(np.sum(w_eff * b_fb) / (np.linalg.norm(w_eff) * np.linalg.norm(b_fb) + 1e-12))

    cos_before = align_cos()
    for _ in range(250):
        x = rng.randn(8)
        y = layer.step(x)
        tgt = w_true @ x
        layer.learn_local(tgt - y)
    cos_after = align_cos()
    assert cos_after > cos_before + 0.35


def test_multilayer_network_and_ops_accounting():
    """DMCNNetwork and DMCNTorchNetwork run cleanly and report positive exact MACs/FLOPs."""
    cfgs = [
        DMCNConfig(d_in=8, d_out=12, n_branches=2, n_spine_scales=2, seed=1),
        DMCNConfig(d_in=12, d_out=3, n_branches=2, n_spine_scales=0, soma_activation="linear", seed=2),
    ]
    net = DMCNNetwork(cfgs)
    pt_net = DMCNTorchNetwork(cfgs, device="cpu", dtype=torch.float32)

    x = np.ones(8, dtype=np.float64) * 0.5
    y = net.step(x)
    _, mse = net.learn_local(np.array([1.0, -0.5, 0.25]))
    assert y.shape == (3,)
    assert np.isfinite(mse)

    y_pt = pt_net(torch.tensor(x, dtype=torch.float32))
    assert tuple(y_pt.shape) == (3,)

    ops = net.total_ops_report()
    r0 = count_layer_ops(cfgs[0])
    assert ops["forward_macs"] > r0.forward_macs
    assert ops["total_flops_per_step"] == ops["forward_flops"] + ops["learn_flops"]
    assert ops["memory_bytes_fp32"] > 0


def test_kwta_and_boids_repulsion_parity():
    """k_wta, center_homeostasis, and boids_repulsion match between NumPy and PyTorch."""
    cfg = DMCNConfig(
        d_in=8,
        d_out=12,
        n_branches=3,
        n_spine_scales=0,
        k_wta=3,
        boids_repulsion=0.15,
        center_homeostasis=True,
        seed=31,
    )
    np_layer = DMCNLayer(cfg)
    pt_layer = DMCNTorchLayer(cfg, device="cpu", dtype=torch.float64)
    pt_layer.load_from_numpy(np_layer)

    rng = np.random.RandomState(31)
    for _ in range(10):
        x = rng.randn(8) + 5.0
        y_np = np_layer.step(x)
        y_pt = pt_layer(torch.tensor(x, dtype=torch.float64)).cpu().numpy()
        assert int(np.count_nonzero(y_np)) <= 3
        np.testing.assert_allclose(y_pt, y_np, atol=1e-6, rtol=1e-5)
        err = rng.randn(12)
        np_layer.learn_local(err)
        pt_layer.learn_local(torch.tensor(err, dtype=torch.float64))


def test_swarm_boids_polarization_and_null_controls():
    """With zero external compass, intact SwarmBoidNetwork aligns via Boids rules while decoupled stays random."""
    cfg = SwarmConfig(n_boids=12, use_dmcn_memory=False, seed=12)
    swarm_intact = SwarmBoidNetwork(cfg)
    swarm_decoupled = SwarmBoidNetwork(cfg)

    pol_intact, pol_dec = [], []
    zero_bearings = np.zeros((12, 2), dtype=np.float64)
    for _ in range(60):
        r_i = swarm_intact.step(zero_bearings, dt=0.2)
        r_d = swarm_decoupled.step(zero_bearings, dt=0.2, decouple_neighbors=True)
        pol_intact.append(r_i["polarization"])
        pol_dec.append(r_d["polarization"])

    assert np.mean(pol_intact[-20:]) > np.mean(pol_dec[-20:]) + 0.25


def test_precision_metaplasticity_order_invariance_and_parity():
    """Sherman-Morrison precision plasticity is strictly order-invariant and matches PyTorch."""
    cfg = DMCNConfig(
        d_in=8,
        d_out=4,
        n_branches=1,
        n_spine_scales=0,
        soma_activation="linear",
        eta_fast=1.0,
        gamma_forget=0.0,
        eta_consol=0.0,
        weight_init_scale=0.0,
        use_precision_plasticity=True,
        precision_ridge=0.5,
        seed=17,
        ablation=AblationConfig(ablate_homeostasis=True),
    )
    rng = np.random.RandomState(17)
    X = rng.randn(30, 8)
    Y = rng.randn(30, 4)

    # Order A: 0..29 vs Order B: 29..0 (zero replay!)
    la = DMCNLayer(cfg)
    lb = DMCNLayer(cfg)
    pt = DMCNTorchLayer(cfg, device="cpu", dtype=torch.float64)
    pt.load_from_numpy(DMCNLayer(cfg))

    for i in range(30):
        ya = la.step(X[i])
        la.learn_local(Y[i] - ya)
        yp = pt(torch.tensor(X[i], dtype=torch.float64))
        pt.learn_local(torch.tensor(Y[i], dtype=torch.float64) - yp)

    for i in reversed(range(30)):
        yb = lb.step(X[i])
        lb.learn_local(Y[i] - yb)

    # Zero-forgetting theorem: W_eff is invariant to presentation order up to float precision!
    np.testing.assert_allclose(la.W_eff, lb.W_eff, atol=1e-10)
    np.testing.assert_allclose(pt.W_eff.cpu().numpy(), la.W_eff, atol=1e-10)


def test_batched_cuda_woodbury_exact_parity_and_general_batch():
    """Batched Woodbury update on CPU/CUDA matches sequential Sherman-Morrison to 1e-10 (FP64)."""
    cfg = DMCNConfig(
        d_in=16,
        d_out=5,
        n_branches=1,
        n_spine_scales=0,
        soma_activation="linear",
        eta_fast=1.0,
        gamma_forget=0.0,
        eta_consol=0.0,
        weight_init_scale=0.0,
        use_precision_plasticity=True,
        precision_ridge=1.5,
        seed=41,
        ablation=AblationConfig(ablate_homeostasis=True),
    )
    rng = np.random.RandomState(41)
    X = rng.randn(50, 16)
    Y = rng.randn(50, 5)

    np_ref = DMCNLayer(cfg)
    for i in range(50):
        y_hat = np_ref.step(X[i])
        np_ref.learn_local(Y[i] - y_hat)

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        pt_b = DMCNTorchLayer(cfg, device=dev, dtype=torch.float64)
        pt_b.load_from_numpy(DMCNLayer(cfg))
        for start in (0, 16, 48):
            end = min(50, start + (16 if start == 0 else 32))
            xb = torch.tensor(X[start:end], device=dev, dtype=torch.float64)
            yb = torch.tensor(Y[start:end], device=dev, dtype=torch.float64)
            pred = pt_b(xb)
            pt_b.learn_local(yb - pred)
        np.testing.assert_allclose(pt_b.W_eff.cpu().numpy(), np_ref.W_eff, atol=1e-10)
        np.testing.assert_allclose(pt_b.P_prec.cpu().numpy(), np_ref.P_prec, atol=1e-10)

        # General 2-layer batched DMCNTorchNetwork (no P_prec) learns a structured nonlinear map on GPU/CPU
        w1_true = rng.randn(16, 8) * 0.6
        w2_true = rng.randn(8, 5) * 0.6
        y_struct = np.tanh(X @ w1_true) @ w2_true
        net_cfgs = [
            DMCNConfig(d_in=16, d_out=24, n_branches=2, n_spine_scales=0, use_nonmonotonic=True, eta_fast=0.5, seed=42),
            DMCNConfig(d_in=24, d_out=5, n_branches=1, n_spine_scales=0, soma_activation="linear", eta_fast=0.5, seed=43),
        ]
        net = DMCNTorchNetwork(net_cfgs, device=dev, dtype=torch.float32)
        x_t = torch.tensor(X, device=dev, dtype=torch.float32)
        y_t = torch.tensor(y_struct, device=dev, dtype=torch.float32)
        mse_start = None
        mse_end = None
        for ep in range(60):
            net(x_t)
            _, mse = net.learn_local(y_t)
            if ep == 0:
                mse_start = mse
            mse_end = mse
        assert np.isfinite(mse_end)
        assert mse_end < 0.5 * mse_start


def test_self_gating_adapts_pathways_without_human_switching():
    """SelfGatingDMCNLayer shifts Hedge gate weight toward homeo/spine under 50x gain shocks."""
    cfg = SelfGatingConfig(d_in=4, d_out=1, d_hid=24, beta_gate=6.0, use_precision=True, seed=51)
    layer = SelfGatingDMCNLayer(cfg)
    rng = np.random.RandomState(51)
    w_true = np.array([1.0, -1.0, 0.5, -0.5])

    for step_i in range(120):
        x_base = rng.randn(4)
        gain = 1.0 if step_i < 40 else (25.0 if step_i < 80 else 0.2)
        y_tgt = float(np.dot(w_true, x_base / (np.sqrt(np.mean(x_base * x_base)) + 1e-3)))
        layer.reset_temporal_state()
        layer.step(x_base * gain, episodic_peak=True)
        layer.learn_local(np.array([y_tgt]))

    g = layer.gate_dict()
    assert (g["homeo"] + g["spine"]) > g["raw"] + 0.30


def test_factorized_bilinear_zero_shot_composition():
    """FactorizedBilinearDMCNLayer generalizes zero-shot on unseen (op, operand) pairs while ablate_bilinear fails."""
    rng = np.random.RandomState(61)
    # 4 operators x 8 distributed 4-D operands (randomly permuted columns so zero index oracle!)
    perm = rng.permutation(8)
    layer_full = FactorizedBilinearDMCNLayer(d_in=8, d_out=4, n_branches=16, ridge=0.05, ablate_bilinear=False, seed=61)
    layer_add = FactorizedBilinearDMCNLayer(d_in=8, d_out=4, n_branches=16, ridge=0.05, ablate_bilinear=True, seed=61)

    ops_M = [np.eye(4), -np.eye(4), np.roll(np.eye(4), 1, axis=0), np.roll(np.eye(4), 2, axis=0)]
    operands = rng.randn(8, 4)
    operands /= (np.linalg.norm(operands, axis=1, keepdims=True) + 1e-8)

    all_pairs = [(o, v) for o in range(4) for v in range(8)]
    # Hold out 8 completely unseen (operator, operand) pairs (2 per operator)
    test_pairs = {(o, (2 * o + k) % 8) for o in range(4) for k in range(2)}
    train_pairs = [p for p in all_pairs if p not in test_pairs]

    def make_xy(o: int, v_idx: int):
        op_oh = np.zeros(4, dtype=np.float64)
        op_oh[o] = 1.0
        val = operands[v_idx]
        x = np.concatenate([op_oh, val])
        return x[perm], ops_M[o] @ val

    for _ in range(12):
        for o, v in train_pairs:
            x, _ = make_xy(o, v)
            layer_full.observe_unsupervised(x)
            layer_add.observe_unsupervised(x)

    for _ in range(4):
        for o, v in train_pairs:
            x, y = make_xy(o, v)
            layer_full.step(x)
            layer_full.learn_local(y)
            layer_add.step(x)
            layer_add.learn_local(y)

    err_full = float(np.mean([np.mean((layer_full.step(make_xy(o, v)[0]) - make_xy(o, v)[1]) ** 2) for o, v in test_pairs]))
    err_add = float(np.mean([np.mean((layer_add.step(make_xy(o, v)[0]) - make_xy(o, v)[1]) ** 2) for o, v in test_pairs]))
    assert err_full < 0.02
    assert err_add > 0.15


def test_unsup_oja_dendritic_focuses_on_moving_pixels():
    """UnsupOjaDendriticLayer increases receptive-field concentration on active moving pixels vs frozen."""
    l_unsup = UnsupOjaDendriticLayer(d_in=64, d_hid=16, k_wta=2, eta_oja=0.15, seed=71, ablate_unsup=False)
    l_frozen = UnsupOjaDendriticLayer(d_in=64, d_hid=16, k_wta=2, eta_oja=0.15, seed=71, ablate_unsup=True)

    # Moving dot only ever appears in top-left 16 pixels (0..15)
    for t in range(120):
        frame = np.zeros(64, dtype=np.float64)
        idx = t % 16
        frame[idx] = 1.0
        l_unsup.step(frame, learn_unsup=True)
        l_frozen.step(frame, learn_unsup=True)

    mass_unsup = float(np.mean(np.sum(l_unsup.W[:, :16] ** 2, axis=1)))
    mass_frozen = float(np.mean(np.sum(l_frozen.W[:, :16] ** 2, axis=1)))
    assert mass_unsup > mass_frozen + 0.30


def test_pdnc_strict_causality_and_zero_autograd():
    """PDNC must be strictly causal online and register zero autograd parameters."""
    cfgs = [
        PDNCConfig(d_in=6, d_out=12, use_transient_spine=True, precision_mode="diagonal", seed=101),
        PDNCConfig(d_in=12, d_out=3, activation="linear", precision_mode="diagonal", weight_init_scale=0.0, seed=102),
    ]
    net_a = PDNCNetwork(cfgs)
    net_b = PDNCNetwork(cfgs)
    pt_net = PDNCTorchNetwork(cfgs, device="cpu", dtype=torch.float64)
    assert len(list(pt_net.parameters())) == 0

    rng = np.random.RandomState(101)
    seq_a = rng.randn(16, 6)
    seq_b = seq_a.copy()
    seq_b[8:] = rng.randn(8, 6) * 10.0

    for t in range(8):
        ya = net_a.step(seq_a[t])
        yb = net_b.step(seq_b[t])
        np.testing.assert_allclose(ya, yb, atol=0.0, rtol=0.0)


def test_pdnc_numpy_torch_parity():
    """PDNCTorchNetwork (CPU & CUDA if available) matches NumPy PDNCNetwork in both diagonal & woodbury modes."""
    rng = np.random.RandomState(111)
    X = rng.randn(24, 8)
    Y = rng.randn(24, 3)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    for p_mode in ("diagonal", "woodbury"):
        cfgs = [
            PDNCConfig(
                d_in=8,
                d_out=10,
                activation="tanh",
                precision_mode=p_mode,
                alpha_null=0.25 if p_mode == "woodbury" else 1.0,
                scale_null_by_batch=(p_mode == "woodbury"),
                eta=0.04,
                seed=111,
            ),
            PDNCConfig(
                d_in=10,
                d_out=3,
                activation="linear",
                precision_mode=p_mode,
                alpha_null=1.0,
                eta=1.0 if p_mode == "woodbury" else 0.04,
                weight_init_scale=0.0,
                seed=112,
            ),
        ]
        np_net = PDNCNetwork(cfgs)
        for start in (0, 8, 16):
            np_net.step(X[start : start + 8])
            np_net.learn_prospective(Y[start : start + 8])

        for dev in devices:
            pt_net = PDNCTorchNetwork(cfgs, device=dev, dtype=torch.float64)
            for start in (0, 8, 16):
                xb = torch.tensor(X[start : start + 8], device=dev, dtype=torch.float64)
                yb = torch.tensor(Y[start : start + 8], device=dev, dtype=torch.float64)
                pt_net(xb)
                pt_net.learn_prospective(yb)
            for l_i in range(2):
                np.testing.assert_allclose(pt_net.layers[l_i].W.cpu().numpy(), np_net.layers[l_i].W, atol=1e-9)
                np.testing.assert_allclose(pt_net.layers[l_i].B_fb.cpu().numpy(), np_net.layers[l_i].B_fb, atol=1e-9)


def test_pdnc_zero_weight_transport_reciprocal_alignment():
    """W and B_fb start uncorrelated (zero weight transport) and align strictly via local endpoint plasticity."""
    layer = PDNCLayer(
        PDNCConfig(
            d_in=16,
            d_out=6,
            activation="linear",
            precision_mode="diagonal",
            eta=0.03,
            lambda_kp=0.02,
            weight_init_scale=1.0,
            fb_init_scale=1.0,
            seed=121,
        )
    )
    cos_init = layer.feedback_alignment_cosine()
    assert abs(cos_init) < 0.30

    rng = np.random.RandomState(121)
    w_target = rng.randn(6, 16) * 0.8
    for _ in range(250):
        xb = rng.randn(16, 16)
        pred = layer.step(xb)
        ds, _ = layer.project_apical((xb @ w_target.T) - pred)
        layer.update_synapses(ds)

    cos_final = layer.feedback_alignment_cosine()
    assert cos_final > 0.85


def test_pdnc_discovers_hidden_nonlinear_features_in_noise():
    """PDNC discovers 2 relevant XOR coordinates buried in 18 noise dims while frozen Layer-1 fails."""
    rng = np.random.RandomState(131)
    n_tr, n_te = 1600, 400
    X_tr = rng.randn(n_tr, 20)
    Y_tr = (np.sign(X_tr[:, 0] * X_tr[:, 1])).reshape(-1, 1)
    X_te = rng.randn(n_te, 20)
    Y_te = (np.sign(X_te[:, 0] * X_te[:, 1])).reshape(-1, 1)

    def build_net(freeze_l1: bool) -> PDNCNetwork:
        return PDNCNetwork([
            PDNCConfig(
                d_in=20,
                d_out=32,
                activation="tanh",
                precision_mode="diagonal",
                eta=0.025,
                weight_decay=0.35,
                prospective_beta=0.8,
                weight_init_scale=0.6,
                bias_init_scale=0.3,
                freeze_weights=freeze_l1,
                seed=131,
            ),
            PDNCConfig(
                d_in=32,
                d_out=1,
                activation="linear",
                precision_mode="diagonal",
                eta=0.02,
                weight_decay=0.1,
                weight_init_scale=0.0,
                fb_init_scale=0.4,
                seed=132,
            ),
        ])

    net_pdnc = build_net(freeze_l1=False)
    net_froz = build_net(freeze_l1=True)

    for _ in range(15):
        perm = rng.permutation(n_tr)
        for s in range(0, n_tr, 32):
            idx = perm[s : s + 32]
            net_pdnc.step(X_tr[idx])
            net_pdnc.learn_prospective(Y_tr[idx])
            net_froz.step(X_tr[idx])
            net_froz.learn_prospective(Y_tr[idx])

    acc_pdnc = float(np.mean(np.sign(net_pdnc.step(X_te)) == Y_te))
    acc_froz = float(np.mean(np.sign(net_froz.step(X_te)) == Y_te))
    sig_energy = float(np.mean(net_pdnc.layers[0].W[:, :2] ** 2))
    noise_energy = float(np.mean(net_pdnc.layers[0].W[:, 2:20] ** 2))

    assert acc_pdnc > 0.88
    assert acc_pdnc > acc_froz + 0.25
    assert sig_energy > 6.0 * noise_energy


def test_pdnc_null_space_protects_prior_subspace():
    """Subspace-projected PDNC learns orthogonal Task-2 features without drifting Task-1 outputs."""
    rng = np.random.RandomState(141)
    # Task 1 lives on dims 0..7; Task 2 lives on dims 8..15
    X1 = np.hstack([rng.randn(256, 8), np.zeros((256, 8))])
    Y1 = (np.sign(X1[:, 0] * X1[:, 1])).reshape(-1, 1)
    X2 = np.hstack([np.zeros((256, 8)), rng.randn(256, 8)])
    Y2 = (np.sign(X2[:, 8] * X2[:, 9])).reshape(-1, 1)

    net = PDNCNetwork([
        PDNCConfig(
            d_in=16,
            d_out=32,
            activation="tanh",
            precision_mode="subspace",
            alpha_null=0.04,
            scale_null_by_batch=True,
            eta=0.03,
            weight_decay=0.05,
            bias_init_scale=0.35,
            prospective_beta=0.8,
            lambda_kp=0.0,
            seed=141,
        ),
        PDNCConfig(
            d_in=32,
            d_out=1,
            activation="linear",
            precision_mode="subspace",
            alpha_null=0.04,
            scale_null_by_batch=True,
            eta=0.03,
            weight_decay=0.05,
            lambda_kp=0.0,
            weight_init_scale=0.0,
            fb_init_scale=0.5,
            seed=142,
        ),
    ])

    for _ in range(18):
        for s in range(0, 256, 32):
            net.step(X1[s : s + 32])
            net.learn_prospective(Y1[s : s + 32])
    acc1_before = float(np.mean(np.sign(net.step(X1)) == Y1))

    for _ in range(18):
        for s in range(0, 256, 32):
            net.step(X2[s : s + 32])
            net.learn_prospective(Y2[s : s + 32])
    acc1_after = float(np.mean(np.sign(net.step(X1)) == Y1))
    acc2_after = float(np.mean(np.sign(net.step(X2)) == Y2))

    assert acc1_before > 0.88
    assert acc2_after > 0.88
    assert acc1_after > acc1_before - 0.05


def test_pdnc_multilayer_rl_discovers_nonlinear_policy():
    """PDNCActorCritic learns a spatial target-tracking RL policy in 12D noise where frozen Layer-1 fails."""

    def run_agent(freeze_l1: bool) -> float:
        ag = PDNCActorCritic(
            enc_cfg=PDNCConfig(
                d_in=12,
                d_out=24,
                activation="tanh",
                precision_mode="diagonal",
                eta=0.015,
                weight_decay=0.05,
                beta_mom=0.90,
                weight_init_scale=0.6,
                bias_init_scale=0.25,
                fb_init_scale=0.4,
                freeze_weights=freeze_l1,
                seed=151,
            ),
            n_actions=2,
            eta_actor=0.02,
            eta_critic=0.03,
            gamma=0.95,
            lam=0.90,
            eps_explore=0.02,
            seed=151,
        )
        ep_rng = np.random.RandomState(151)
        for _ in range(700):
            ag.reset_episode()
            # 3-step pursuit episode buried in 10 high-variance irrelevant distractor dims (scale 3.0)
            target = float(ep_rng.choice([-1.0, 1.0])) * ep_rng.uniform(0.5, 1.2)
            pos = 0.0
            for t in range(3):
                obs = ep_rng.randn(12) * 3.0
                obs[0], obs[1] = target, pos
                act, v_c = ag.step_policy(obs, greedy=False)
                pos += 0.5 if act == 1 else -0.5
                done = t == 2
                rew = (1.0 if abs(target - pos) < 0.8 else -1.0) if done else 0.0
                v_n = 0.0
                if not done:
                    obs_n = ep_rng.randn(12) * 3.0
                    obs_n[0], obs_n[1] = target, pos
                    v_n = ag.value_only(obs_n)
                ag.apply_td_update(v_c, rew, v_n, done)

        eval_rng = np.random.RandomState(999)
        correct = 0
        for _ in range(150):
            ag.reset_episode()
            target = float(eval_rng.choice([-1.0, 1.0])) * eval_rng.uniform(0.5, 1.2)
            pos = 0.0
            for _ in range(3):
                obs = eval_rng.randn(12) * 3.0
                obs[0], obs[1] = target, pos
                act, _ = ag.step_policy(obs, greedy=True)
                pos += 0.5 if act == 1 else -0.5
            if abs(target - pos) < 0.8:
                correct += 1
        return correct / 150.0

    acc_learn = run_agent(freeze_l1=False)
    acc_frozen = run_agent(freeze_l1=True)
    assert acc_learn > 0.85
    assert acc_learn > acc_frozen + 0.15


def test_gram_cholesky_parity_and_sparse_ternary_and_patch_mask():
    """Verify Gram-Cholesky exact parity vs Woodbury, 95%-sparse ternary weights, and local RF mask invariance."""
    from neuron_v2.dmcn import DMCNConfig, DMCNTorchLayer, PDNCConfig, PDNCLayer, make_patch_receptive_field_mask

    rng = np.random.RandomState(2026)
    X = rng.randn(128, 32)
    Y = rng.randn(128, 6)

    cfg_w = DMCNConfig(
        d_in=32, d_out=6, n_branches=1, n_spine_scales=0, soma_activation="linear",
        eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0, weight_init_scale=0.0,
        use_precision_plasticity=True, precision_ridge=5.0, precision_update_mode="woodbury", seed=1,
    )
    cfg_g = DMCNConfig(
        d_in=32, d_out=6, n_branches=1, n_spine_scales=0, soma_activation="linear",
        eta_fast=1.0, gamma_forget=0.0, eta_consol=0.0, weight_init_scale=0.0,
        use_precision_plasticity=True, precision_ridge=5.0, precision_update_mode="gram_cholesky", seed=1,
    )
    cfg_w.ablation.ablate_homeostasis = True
    cfg_g.ablation.ablate_homeostasis = True

    lw = DMCNTorchLayer(cfg_w, device="cpu", dtype=torch.float64)
    lg = DMCNTorchLayer(cfg_g, device="cpu", dtype=torch.float64)
    xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(Y, dtype=torch.float64)
    for s in range(0, 128, 32):
        pred_w = lw(xt[s : s + 32])
        lw.learn_local(yt[s : s + 32] - pred_w)
        pred_g = lg(xt[s : s + 32])
        lg.learn_local(yt[s : s + 32] - pred_g)
    lg.consolidate_gram()
    np.testing.assert_allclose(lg.W_eff.numpy(), lw.W_eff.numpy(), atol=1e-10)

    cfg_t = DMCNConfig(
        d_in=128, d_out=64, n_branches=1, n_spine_scales=0,
        soma_activation="tanh", use_sparse_ternary=True, ternary_density=0.05, seed=7,
    )
    lt = DMCNTorchLayer(cfg_t, device="cpu", dtype=torch.float32)
    w_arr = lt.W_eff.numpy()
    nz_frac = float(np.mean(w_arr != 0.0))
    assert 0.02 < nz_frac < 0.09
    uniq = np.unique(np.abs(w_arr))
    assert len(uniq) <= 2  # {0.0, scale}

    mask = make_patch_receptive_field_mask(8, 8, patch_size=4, stride=2, compartments_per_patch=2)
    pdnc = PDNCLayer(PDNCConfig(d_in=64, d_out=mask.shape[0], use_transient_spine=True, receptive_field_mask=mask, seed=9))
    for _ in range(5):
        pdnc.step(rng.randn(64))
        pdnc.accumulate_eligibility(rng.randn(mask.shape[0]))
        pdnc.apply_rpe(1.0, consolidate=True)
    np.testing.assert_array_equal(pdnc.W[mask == 0.0], 0.0)


