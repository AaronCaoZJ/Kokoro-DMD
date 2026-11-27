"""CPU unit tests for KokoroDMD2ODE: a toy network stands in for BAGEL to check each variant's
rollout, gradient flow, detach placement, noising construction, train/inference consistency and
exit-step broadcast.

    PYTHONPATH=$PWD python -m pytest -q kokoro_ode/tests/test_ode_methods.py   # run from the repo root
"""

from __future__ import annotations

import os
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fastgen.configs.config import SampleTConfig
from fastgen.methods.distribution_matching import kokoro_dpdmd as dpdmd_mod
from fastgen.methods.distribution_matching import kokoro_ode_common as common_mod
from fastgen.methods.distribution_matching import kokoro_pdd as pdd_mod
from fastgen.methods.distribution_matching.kokoro_dmd2_ode import KokoroDMD2ODE
from fastgen.methods.distribution_matching.kokoro_ode_common import shifted_grid
from fastgen.networks.network import FastGenNetwork

D = 64          # patch_dim
H = W = 8       # latent 16×8×8 → 4×4 = 16 tokens
T_LIST = [0.999, 0.937, 0.833, 0.624, 0.0]


class ToyNet(FastGenNetwork):
    """v = tanh(x·A) + b·t + c·role_bias; records every call (role, t, grad enabled); outputs can retain_grad."""

    def __init__(self, seed: int, role_bias: float = 0.0, pdd_grid_steps: int = 0):
        super().__init__(net_pred_type="v", schedule_type="rf")
        g = torch.Generator().manual_seed(seed)
        self.A = torch.nn.Parameter(torch.randn(D, D, generator=g) / D**0.5)
        self.b = torch.nn.Parameter(torch.randn(D, generator=g) * 0.1)
        self.pdd_grid_steps = pdd_grid_steps
        if pdd_grid_steps:      # each head = shared output + a learnable bias c_k (all 0 after init = N identical heads)
            self.c = torch.nn.Parameter(torch.randn(pdd_grid_steps, D, generator=g) * 0.1)
        self.role_bias = role_bias
        self.calls: list[dict] = []
        self.keep_outputs = False
        self.outputs: list[torch.Tensor] = []

    def init_pdd_heads_from_llm2vae(self):
        with torch.no_grad():
            self.c.zero_()

    def forward(self, x_t, t, condition=None, role="student", fwd_pred_type="v", pdd=False, **kw):
        tv = t.reshape(-1)
        assert bool((tv.max() == tv.min()).item()), "single-timestep invariant"
        assert x_t.dtype == self.A.dtype, f"x must already be in network precision when it enters the net, got {x_t.dtype}"
        bias = self.role_bias if role == "teacher" else 0.0
        v = torch.tanh(x_t @ self.A) + self.b * tv[0] + bias
        self.calls.append({"role": role, "t": float(tv[0]), "grad": torch.is_grad_enabled(),
                           "pred": "pdd" if pdd else fwd_pred_type})
        if pdd:
            assert role == "student" and self.pdd_grid_steps
            u = v.unsqueeze(1) + self.c.unsqueeze(0)          # (N_vae, N, D)
            if self.keep_outputs and u.requires_grad:
                u.retain_grad()
                self.outputs.append(u)
            return u
        if self.keep_outputs and v.requires_grad:
            v.retain_grad()
            self.outputs.append(v)
        if fwd_pred_type == "x0":
            return x_t - t.unsqueeze(-1) * v
        return v

    # KokoroNetwork interface stubs
    def apply_trainable_param_patterns(self, pats):
        self.requires_grad_(True)

    def build_inference_condition(self, sample, device, cfg_text_scale=1.0, cfg_img_scale=1.0):
        c, h, w = sample["vae_latent"].shape
        return {"task_type": "t2i", "patchified_vae_latent_shapes": [(h // 2, w // 2)],
                "cfg_text_scale": cfg_text_scale, "cfg_img_scale": cfg_img_scale}


def make_method(precision=torch.float32, **switches):
    st = SampleTConfig()
    st.t_list = list(switches.pop("t_list", T_LIST))
    st.time_dist_type = "shifted"
    st.min_t, st.max_t = 0.001, 0.999
    cfg = types.SimpleNamespace(
        student_sample_steps=len(st.t_list) - 1, student_update_freq=5, sample_t_cfg=st,
        student_first_step_weight=1.0, student_sample_type="ode", gan_loss_weight_gen=0.0,
        latent_patch_size=2, sample_logging_iter=0, logging_iter=50, guidance_scale=4.0, cfg_img_teacher=2.0,
        trainable_param_patterns=["moe_gen"], gan_r1_reg_weight=0.0, gan_use_same_t_noise=True,
    )
    for k, v in switches.items():
        setattr(cfg, k, v)
    KokoroDMD2ODE._validate_switches(cfg)
    m = KokoroDMD2ODE.__new__(KokoroDMD2ODE)
    torch.nn.Module.__init__(m)
    m.config = cfg
    m.device = torch.device("cpu")
    m.precision = precision
    m.precision_amp_infer = None
    m.use_ema = []
    n_pdd = int(getattr(cfg, "pdd_grid_steps", 0)) if getattr(cfg, "pdd", False) else 0
    m.net = ToyNet(0, pdd_grid_steps=n_pdd).to(precision)
    m.teacher = ToyNet(1, role_bias=0.05).to(precision).requires_grad_(False)
    m.fake_score = ToyNet(2).to(precision)
    return m


def sample_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"vae_latent": torch.randn(16, H, W, generator=g), "text": "x", "task_type": "t2i"}]


def cond():
    return {"task_type": "t2i", "patchified_vae_latent_shapes": [(H // 2, W // 2)],
            "cfg_text_scale": 4.0, "cfg_img_scale": 2.0}


def n_tok():
    return (H // 2) * (W // 2)


def reset(m):
    for net in (m.net, m.teacher, m.fake_score):
        net.calls.clear()
        net.outputs.clear()
        net.zero_grad(set_to_none=True)


def force_exit(m, k):
    m._sample_exit_step = lambda n_steps: k


# ── 1. ODE = Euler = DDIM, training rollout = inference ──────────────────────

def test_rollout_is_euler_and_matches_inference():
    m = make_method(student_rollout="ode")
    torch.manual_seed(0)
    noise = torch.randn(n_tok(), D)
    t = m._student_t_values()
    # Hand-written reference: x = t0·ε; x += (t_{i+1}−t_i)·v; finally x0 = x − t_3·v
    x = noise * t[0]
    with torch.no_grad():
        for i in range(3):
            x = x + (t[i + 1] - t[i]) * m.net(x, torch.full((n_tok(),), t[i]), condition=cond())
        ref = x - t[3] * m.net(x, torch.full((n_tok(),), t[3]), condition=cond())
        x3 = m._ode_rollout_to(noise, t, 3, cond())
        _, x0_train = m._exit_forward(x3, t[3], cond())
    x0_infer = m.packed_generator_fn(m.net, noise, cond(), 4, "ode", T_LIST)
    torch.testing.assert_close(x0_train, ref)
    torch.testing.assert_close(x0_infer, ref)
    # The parent class's sampling loop (x0 → implied ε → forward process) is the same integrator as Euler in fp32
    with torch.no_grad():
        old = KokoroDMD2._packed_student_sample_loop(m, noise * t[0], torch.tensor(t), cond(), "ode")
    torch.testing.assert_close(old, ref, rtol=1e-4, atol=1e-5)


from fastgen.methods.distribution_matching.kokoro_dmd2 import KokoroDMD2  # noqa: E402


def test_bf16_t_quantization_and_anchor_alignment():
    m = make_method(precision=torch.bfloat16, student_rollout="ode")
    t = m._student_t_values()
    assert t == [1.0, 0.9375, 0.83203125, 0.625, 0.0]
    grid = sorted({m._q(v) for v in shifted_grid(30, 3.0, t[0])}, reverse=True)
    assert grid[5] == 0.9375 == t[1], "DP-DMD anchor K=5 should land exactly on the student's first-step landing point"
    g2 = sorted({m._q(v) for v in shifted_grid(40, 3.0, t[0], include=t)}, reverse=True)
    assert all(v in g2 for v in t) and g2[0] == t[0] and g2[-1] == 0.0


# ── 2. B (ODE rollout): only the exit step carries gradient ─────────────────────

@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_B_gradient_only_through_exit_step(k):
    m = make_method(student_rollout="ode")
    force_exit(m, k)
    reset(m)
    noise = torch.randn(n_tok(), D)
    real = torch.randn(n_tok(), D)
    loss_map, out = m._rollout_student_step(noise, real, cond())
    student = [c for c in m.net.calls if c["role"] == "student"]
    assert [c["grad"] for c in student] == [False] * k + [True]
    assert [c["t"] for c in student] == pytest.approx(m._student_t_values()[: k + 1])
    assert [c["grad"] for c in m.teacher.calls] == [False] and [c["grad"] for c in m.fake_score.calls] == [False]
    loss_map["total_loss"].backward()
    assert m.net.A.grad is not None and m.net.A.grad.abs().sum() > 0
    assert m.teacher.A.grad is None and m.fake_score.A.grad is None
    assert torch.isfinite(loss_map["total_loss"])


# ── 3. C (trajectory noise): the comparison noise = the trajectory's starting noise ──

def test_C_uses_trajectory_noise():
    m = make_method(student_rollout="ode", dmd_noise="trajectory")
    force_exit(m, 2)
    captured = {}
    orig = m._dmd_compare

    def spy(gen, x_t, t_vsd, c):
        captured.update(gen=gen.detach(), x_t=x_t, tau=float(t_vsd[0]))
        return orig(gen, x_t, t_vsd, c)

    m._dmd_compare = spy
    noise = torch.randn(n_tok(), D)
    m._rollout_student_step(noise, torch.randn(n_tok(), D), cond())
    tau = captured["tau"]
    torch.testing.assert_close(captured["x_t"], (1 - tau) * captured["gen"] + tau * noise)
    # fake step: target v = ε_traj − x̂0
    reset(m)
    loss_map, out = m._rollout_fake_step(noise, cond())
    assert torch.isfinite(loss_map["total_loss"])


# ── 4. D (DP-DMD): detach after step 1, DMD gradient only through steps 2..N, diversity term only on step 1 ──

@pytest.mark.parametrize("grad_steps", ["after_first", "last"])
def test_D_detach_and_gradient_routing(grad_steps):
    m = make_method(student_rollout="ode", dpdmd=True, dpdmd_grad_steps=grad_steps, dpdmd_k_anchor=5,
                    dpdmd_teacher_steps=30)
    m.net.keep_outputs = True
    reset(m)
    noise = torch.randn(n_tok(), D)
    captured = {}
    real_mse = torch.nn.functional.mse_loss

    def spy_mse(a, b, *args, **kw):
        out = real_mse(a, b, *args, **kw)
        captured.setdefault("div", out)            # the first mse_loss in the student step is the diversity term
        return out

    dpdmd_mod.F.mse_loss = spy_mse
    try:
        loss_map, out = m._dpdmd_student_step(noise, torch.randn(n_tok(), D), cond())
    finally:
        dpdmd_mod.F.mse_loss = real_mse
    student = [c for c in m.net.calls if c["role"] == "student"]
    expect = [True, True, True, True] if grad_steps == "after_first" else [True, False, False, True]
    assert [c["grad"] for c in student] == expect
    assert sum(1 for c in m.teacher.calls if c["grad"]) == 0
    assert len(m.teacher.calls) == 5 + 1, "teacher: 5 anchor steps + 1 VSD comparison"
    vs = m.net.outputs          # v's with gradient: [v1, (v2, v3,) v4]
    # Backprop DMD only: v1 gets no gradient
    loss_map["vsd_loss"].backward(retain_graph=True)
    assert vs[0].grad is None or vs[0].grad.abs().sum() == 0
    for v in vs[1:]:
        assert v.grad is not None and v.grad.abs().sum() > 0
    for v in vs:
        v.grad = None
    # Backprop the diversity term only: only v1 gets gradient
    torch.testing.assert_close(captured["div"].detach(), loss_map["div_loss"])
    captured["div"].backward(retain_graph=True)
    assert vs[0].grad.abs().sum() > 0 and all(v.grad is None for v in vs[1:])
    # total loss = vsd + λ·div
    torch.testing.assert_close(
        loss_map["total_loss"], loss_map["vsd_loss"] + 0.05 * loss_map["div_loss"], rtol=1e-5, atol=1e-6
    )


def test_D_anchor_target():
    m = make_method(student_rollout="ode", dpdmd=True, dpdmd_k_anchor=5, dpdmd_teacher_steps=30)
    noise = torch.randn(n_tok(), D)
    t = m._student_t_values()
    grid = sorted({m._q(v) for v in shifted_grid(30, 3.0, t[0])}, reverse=True)
    x = noise * t[0]
    with torch.no_grad():
        for i in range(5):
            x = x + (grid[i + 1] - grid[i]) * m.teacher(x, torch.full((n_tok(),), grid[i]), condition=cond(),
                                                        role="teacher")
    v_target = (noise * t[0] - x) / (t[0] - grid[5])
    captured = {}
    real_mse = torch.nn.functional.mse_loss

    def spy_mse(a, b, *args, **kw):
        if "target" not in captured:
            captured["target"] = b
        return real_mse(a, b, *args, **kw)

    dpdmd_mod.F.mse_loss = spy_mse
    try:
        m._dpdmd_student_step(noise, torch.randn(n_tok(), D), cond())
    finally:
        dpdmd_mod.F.mse_loss = real_mse
    torch.testing.assert_close(captured["target"], v_target)


def test_D_fake_uses_final_output():
    m = make_method(student_rollout="ode", dpdmd=True)
    m._sample_exit_step = lambda n: (_ for _ in ()).throw(AssertionError("the D fake step must not sample an exit step"))
    reset(m)
    noise = torch.randn(n_tok(), D)
    loss_map, out = m._rollout_fake_step(noise, cond())
    assert len([c for c in m.net.calls if c["role"] == "student"]) == 4
    assert all(not c["grad"] for c in m.net.calls)
    torch.testing.assert_close(out["gen_packed"], m.packed_generator_fn(m.net, noise, cond(), 4, "ode", T_LIST))
    loss_map["total_loss"].backward()
    assert m.fake_score.A.grad is not None and m.net.A.grad is None


# ── 5. TDM: landing point, current-step interval, mixed noise, importance weight ──

def test_TDM_landing_construction():
    m = make_method(student_rollout="ode", dmd_target="landing")
    t = m._student_t_values()
    torch.manual_seed(0)
    x_k = torch.randn(4096, D) * t[1]
    v = torch.randn(4096, D)
    x0 = x_k - t[1] * v
    for _ in range(20):
        x_tau, t_tok, info = m._tdm_noised(x_k, v, x0, t[1], t[2])
        tau = info["tau"]
        assert t[2] <= tau < t[1] or tau == t[1]
        torch.testing.assert_close(x_tau, (1 - tau) * x0 + tau * info["mixed"], rtol=1e-4, atol=1e-4)
        assert float(t_tok[0]) == tau
    # landing point = one step along the student's own velocity
    eps_hat = x_k + (1 - t[1]) * v
    torch.testing.assert_close((1 - t[2]) * x0 + t[2] * eps_hat, x_k + (t[2] - t[1]) * v, rtol=1e-5, atol=1e-5)
    # With ε̂ and z both standard Gaussian, the mixed noise has variance ≈ 1 and weight ≈ 1
    x0g = torch.randn(4096, D)
    epsg = torch.randn(4096, D)
    x_kg = (1 - t[1]) * x0g + t[1] * epsg
    vg = epsg - x0g
    _, _, info = m._tdm_noised(x_kg, vg, x0g, t[1], t[2])
    assert abs(info["mixed"].var().item() - 1.0) < 0.02
    assert abs(info["is_weight"].item() - 1.0) < 0.02
    # Last step: landing point = x̂0, m = z, weight = 1, τ ∈ [tdm_min_t, t_{N−1})
    _, _, info = m._tdm_noised(x_kg, vg, x0g, t[3], 0.0)
    assert 0.02 <= info["tau"] <= t[3] and info["is_weight"].item() == pytest.approx(1.0)


def test_TDM_student_and_fake_steps():
    m = make_method(student_rollout="ode", dmd_target="landing")
    force_exit(m, 1)
    reset(m)
    noise = torch.randn(n_tok(), D)
    loss_map, out = m._rollout_student_step(noise, torch.randn(n_tok(), D), cond())
    t = m._student_t_values()
    tau = float(loss_map["tdm_tau"])
    assert t[2] <= tau <= t[1]
    assert m.teacher.calls[0]["t"] == pytest.approx(tau)
    loss_map["total_loss"].backward()
    assert m.net.A.grad is not None and m.fake_score.A.grad is None
    reset(m)
    loss_map, _ = m._rollout_fake_step(noise, cond())
    assert "tdm_is_weight" in loss_map and torch.isfinite(loss_map["total_loss"])
    loss_map["total_loss"].backward()
    assert m.fake_score.A.grad is not None and m.net.A.grad is None


# ── 6. B/C fake step: re-sample the exit step so samples match the student step's distribution ──

@pytest.mark.parametrize("fake_input,expect_k", [("matched", 2), ("final", 3)])
def test_B_fake_step_exit(fake_input, expect_k):
    m = make_method(student_rollout="ode", fake_input=fake_input)
    force_exit(m, 2)
    reset(m)
    noise = torch.randn(n_tok(), D)
    loss_map, out = m._rollout_fake_step(noise, cond())
    assert int(loss_map["rollout_exit_k_fake"]) == expect_k
    assert len(m.net.calls) == expect_k + 1 and all(not c["grad"] for c in m.net.calls)
    x_k = m._ode_rollout_to(noise, m._student_t_values(), expect_k, cond())
    with torch.no_grad():
        _, x0 = m._exit_forward(x_k, m._student_t_values()[expect_k], cond())
    torch.testing.assert_close(out["gen_packed"], x0)


# ── 7. ODE-init ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("points", ["all", "random"])
def test_ode_init(points):
    m = make_method(ode_init=True, student_update_freq=1, ode_init_points=points, ode_init_teacher_steps=40)
    force_exit(m, 2)
    reset(m)
    loss_map, out = m.single_train_step(sample_batch(), iteration=3)
    t = m._student_t_values()
    grid = sorted({m._q(v) for v in shifted_grid(40, 3.0, t[0], include=t)}, reverse=True)
    assert len(m.teacher.calls) == len(grid) - 1 and all(not c["grad"] for c in m.teacher.calls)
    student = [c for c in m.net.calls if c["role"] == "student"]
    if points == "all":
        assert [c["t"] for c in student] == pytest.approx(t[:4])
    else:
        assert len(student) == 1 and any(student[0]["t"] == pytest.approx(v) for v in t[:4])
    assert all(c["grad"] for c in student)
    loss_map["total_loss"].backward()
    assert m.net.A.grad is not None and m.fake_score.A.grad is None and m.teacher.A.grad is None


# ── 8. End-to-end single_train_step (student step + fake step for every variant) ──

# A = parent DMD2 path (real-noised), B = ODE rollout, C = B + trajectory noise,
# D = DP-DMD, D_xt = D with x_t-normalized VSD, TDM = landing-point target
ARMS = {
    "A": dict(student_rollout="real_noised"),
    "B": dict(student_rollout="ode"),
    "C": dict(student_rollout="ode", dmd_noise="trajectory"),
    "D": dict(student_rollout="ode", dpdmd=True),
    "D_xt": dict(student_rollout="ode", dpdmd=True, vsd_norm="xt"),
    "TDM": dict(student_rollout="ode", dmd_target="landing"),
}


@pytest.mark.parametrize("arm", list(ARMS))
def test_single_train_step_all_arms(arm):
    m = make_method(**ARMS[arm])
    m.config.sample_logging_iter = 5
    for it in (5, 6):          # 5 = student step (with 4-step preview), 6 = fake step
        reset(m)
        loss_map, out = m.single_train_step(sample_batch(it), iteration=it)
        assert torch.isfinite(loss_map["total_loss"]), (arm, it, loss_map)
        loss_map["total_loss"].backward()
        if it == 5:
            assert m.net.A.grad is not None and m.fake_score.A.grad is None
            assert "student_4nfe" in out["kokoro_x0_vis"]
        else:
            assert m.fake_score.A.grad is not None and m.net.A.grad is None


def test_A_scores_rebuilt_in_fp32():
    """Variant A (parent-class path): fake / teacher predict v and x0 is rebuilt in fp32 (as in the ODE
    path's _fake_teacher_x0), instead of the network emitting x0 in bf16 and building the pseudo-target in bf16."""
    m = make_method(precision=torch.bfloat16, student_rollout="real_noised")
    reset(m)
    loss_map, out = m.single_train_step(sample_batch(5), iteration=5)
    assert torch.isfinite(loss_map["total_loss"])
    assert {(c["role"], c["pred"]) for c in m.fake_score.calls + m.teacher.calls} == {("fake", "v"), ("teacher", "v")}
    loss_map["total_loss"].backward()
    assert m.net.A.grad is not None and m.fake_score.A.grad is None


# ── 9. Switch validation ─────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    dict(student_rollout="real_noised", dmd_noise="trajectory"),
    dict(student_rollout="real_noised", dpdmd=True),
    dict(student_rollout="ode", dmd_target="landing", dmd_noise="trajectory"),
    dict(student_rollout="ode", dmd_target="landing", dpdmd=True),
    dict(student_rollout="ode", gan_loss_weight_gen=1e-3),
    dict(student_rollout="ode", student_sample_type="sde"),
    dict(student_rollout="sde"),
    dict(ode_init=True),                                  # STUDENT_UPDATE_FREQ=1 not set
    dict(student_rollout="ode", dpdmd=True, dpdmd_k_anchor=31),
    dict(student_rollout="ode", student_update_freq=1),   # with a fake score, freq=1 means fake is never updated
    dict(student_rollout="real_noised", student_update_freq=1),
])
def test_invalid_switches(bad):
    with pytest.raises((ValueError, NotImplementedError)):
        make_method(**bad)


# ── 10. Exit step: sampled on rank 0, identical on all ranks (2-process gloo) ──

def _exit_worker(rank, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    common_mod.world_size = lambda: dist.get_world_size()   # fastgen's world_size() is always 1 on CPU
    torch.manual_seed(1000 + rank)                       # different RNG state per rank
    m = make_method(student_rollout="ode")
    ks = [m._sample_exit_step(4) for _ in range(64)]
    out[rank] = ks
    dist.destroy_process_group()


def test_exit_step_broadcast():
    out = mp.Manager().dict()
    mp.spawn(_exit_worker, args=(29611, out), nprocs=2, join=True)
    assert out[0] == out[1]
    assert set(out[0]) == {0, 1, 2, 3}


# ── 11. PDD ───────────────────────────────────────────────────────────────────

from fastgen.methods.distribution_matching.kokoro_ode_common import ode_sample  # noqa: E402
from fastgen.methods.distribution_matching.kokoro_pdd import pdd_advance, pdd_grid, pdd_sample  # noqa: E402

PDD_KW = dict(pdd=True, pdd_grid_steps=32, pdd_shift=5.0, student_update_freq=1)


def test_pdd_grid_matches_student_table():
    g = pdd_grid(128, 5.0)
    assert len(g) == 129 and g[0] == 1.0 and g[-1] == 0.0 and all(a > b for a, b in zip(g[:-1], g[1:]))
    # The 4-NFE block boundaries = the shift-5 4-step table ≈ T_LIST (0.999→1, 0.937→0.9375, 0.833, 0.624→0.625)
    assert g[0::32] == pytest.approx([1.0, 0.9375, 5 * 0.5 / 3, 0.625, 0.0])


def test_pdd_advance_and_fused_sampler():
    torch.manual_seed(0)
    g = pdd_grid(16, 5.0)
    x = torch.randn(n_tok(), D)
    u = torch.randn(n_tok(), 16, D)
    ref = x.clone()
    for k in range(4, 9):
        ref = ref + (g[k + 1] - g[k]) * u[:, k]
    torch.testing.assert_close(pdd_advance(x, u, g, 4, 5), ref)
    torch.testing.assert_close(pdd_advance(x, u, g, 4, 0), x)
    # With identical heads (right after init), a block step = Euler on the coarse grid: pdd_sample == ode_sample(grid[0::L])
    m = make_method(**PDD_KW)
    m.net.init_pdd_heads_from_llm2vae()
    noise = torch.randn(n_tok(), D)
    grid = pdd_grid(32, 5.0)
    a = pdd_sample(m.net, noise, grid, 8, cond(), torch.float32)
    b = ode_sample(m.net, noise, grid[0::8], cond(), torch.float32)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    # In PDD mode packed_generator_fn uses block-step sampling, NFE = student_sample_steps
    reset(m)
    out = m.packed_generator_fn(m.net, noise, cond(), 4, "ode", T_LIST)
    torch.testing.assert_close(out, a)
    assert len(m.net.calls) == 4 and all(c["pred"] == "pdd" for c in m.net.calls)


@pytest.mark.parametrize("solver", ["midpoint", "euler"])
@pytest.mark.parametrize("n_start", [0, 4, 12])
def test_pdd_step(solver, n_start):
    m = make_method(**PDD_KW, pdd_solver=solver)       # N=32 → L_min=4, L_max=16
    m.net.keep_outputs = True
    m._broadcast_int = lambda v: n_start
    reset(m)
    torch.manual_seed(1)
    ks_seen = {}
    real_perm = torch.randperm

    def spy_perm(n, *a, **kw):
        out = real_perm(n, *a, **kw)
        ks_seen["perm"] = out
        return out

    pdd_mod.torch.randperm = spy_perm
    try:
        loss_map, out = m.single_train_step(sample_batch(), iteration=3)
    finally:
        pdd_mod.torch.randperm = real_perm
    grid = pdd_grid(32, 5.0)
    student = [c for c in m.net.calls if c["role"] == "student"]
    assert [c["grad"] for c in student] == [False] * (n_start // 4) + [True]
    assert [c["t"] for c in student] == pytest.approx([grid[i] for i in range(0, n_start + 1, 4)])
    n_k = 2 if solver == "euler" else 1
    ks = (ks_seen["perm"][:n_k] + n_start).tolist()
    assert all(n_start <= k < min(n_start + 16, 32) for k in ks)
    teacher_t = [c["t"] for c in m.teacher.calls]
    if solver == "midpoint":
        k = ks[0]
        assert teacher_t == pytest.approx([grid[k], 0.5 * (grid[k] + grid[k + 1])])
    else:
        assert teacher_t == pytest.approx([grid[k] for k in ks])
    assert all(not c["grad"] for c in m.teacher.calls)
    # Where the gradient goes (the loss value is checked by hand in test_pdd_loss_value_midpoint)
    u = m.net.outputs[-1]
    loss_map["total_loss"].backward()
    grads = u.grad.abs().sum(dim=(0, 2))
    assert set(torch.nonzero(grads).flatten().tolist()) == set(ks), "only the sampled heads k receive gradient"
    assert m.net.c.grad is not None and m.teacher.A.grad is None
    assert torch.isfinite(loss_map["total_loss"])


def test_pdd_loss_value_midpoint():
    m = make_method(**PDD_KW, pdd_solver="midpoint")
    m._broadcast_int = lambda v: 0
    grid = pdd_grid(32, 5.0)
    torch.manual_seed(5)
    real_perm = torch.randperm
    pdd_mod.torch.randperm = lambda n, *a, **kw: torch.arange(n).roll(-3)   # k = 3
    captured = {}
    real_heads = m._pdd_heads

    def spy_heads(x, t, c):
        out = real_heads(x, t, c)
        captured.setdefault("x_n", x.detach().clone())
        captured["u"] = out
        return out

    m._pdd_heads = spy_heads
    try:
        loss_map, _ = m.single_train_step(sample_batch(), iteration=3)
    finally:
        pdd_mod.torch.randperm = real_perm
    u = captured["u"].detach()
    x_k = captured["x_n"] + sum((grid[l + 1] - grid[l]) * u[:, l] for l in range(0, 3))
    h = grid[4] - grid[3]
    tt = lambda t: torch.full((n_tok(),), t)
    with torch.no_grad():
        v1 = m.teacher(x_k, tt(grid[3]), condition=cond(), role="teacher")
        v2 = m.teacher(x_k + 0.5 * h * v1, tt(grid[3] + 0.5 * h), condition=cond(), role="teacher")
    torch.testing.assert_close(loss_map["total_loss"], torch.nn.functional.mse_loss(u[:, 3], v2))


# ── 12. TDM losses checked term by term against the reference implementation ──

@pytest.mark.parametrize("c", [1e-3, 0.0])
def test_tdm_generator_loss_formula(c):
    m = make_method(student_rollout="ode", dmd_target="landing", tdm_huber_c=c)
    torch.manual_seed(0)
    gen = torch.randn(n_tok(), D, requires_grad=True)
    real, fake = torch.randn(n_tok(), D), torch.randn(n_tok(), D)
    loss, _ = m._tdm_generator_loss(gen, real, fake)
    w = (gen.detach() - real).abs().mean()
    coop = gen.detach() + (real - fake)
    if c > 0:
        ref = ((torch.sqrt((gen - coop) ** 2 + c * c) - c) / w).mean()
    else:
        ref = (torch.nn.functional.mse_loss(gen, coop, reduction="none") / w).mean()
    torch.testing.assert_close(loss, ref)
    loss.backward()
    # Gradient direction: pushes x̂0 toward (real − fake)
    assert (gen.grad * (real - fake)).sum() < 0


def test_tdm_fake_weight_floor():
    """f = 1 is the default weight, f = 0 is equivalent to the reference demo's weight, and f = 0.25 matches
    the demo for τ ≤ 0.5 and is floored at 0.25 above that."""
    from fastgen.methods.distribution_matching.kokoro_tdm import tdm_fake_weight
    for tau in (0.07, 0.1, 0.3, 0.5, 0.75, 0.9, 0.95, 0.989):
        snr = ((1 - tau) / tau) ** 2
        assert tdm_fake_weight(tau, 1.0) == pytest.approx(min(1.0, 5.0 / snr))
        assert tdm_fake_weight(tau, 0.0) == pytest.approx(min((1 - tau) ** 2, 5 * tau ** 2))     # = τ²·min(SNR, 5)
        assert tdm_fake_weight(tau, 0.0) == pytest.approx(tau ** 2 * min(snr, 5.0))
        if tau <= 0.5:
            assert tdm_fake_weight(tau, 0.25) == pytest.approx(tdm_fake_weight(tau, 0.0))
        else:
            assert tdm_fake_weight(tau, 0.25) == pytest.approx(0.25)


@pytest.mark.parametrize("floor", [1.0, 0.25, 0.0])
def test_tdm_fake_loss_formula(floor):
    """L = is_weight · min(1, 5/SNR)·max((1−τ)², f) · ‖v_fake − (m − x̂0)‖²; f = 1 is the default."""
    m = make_method(student_rollout="ode", dmd_target="landing", tdm_fake_w_floor=floor)
    force_exit(m, 1)
    captured = {}
    real_noised = m._tdm_noised

    def spy(*a):
        out = real_noised(*a)
        captured.update(x_t=out[0], info=out[2], x0=a[2])
        return out

    m._tdm_noised = spy
    reset(m)
    loss_map, _ = m._rollout_fake_step(torch.randn(n_tok(), D), cond())
    info, tau = captured["info"], captured["info"]["tau"]
    with torch.no_grad():
        fv = m.fake_score(captured["x_t"], torch.full((n_tok(),), tau), condition=cond(), fwd_pred_type="v", role="fake")
    w = min(1.0, 5.0 * (tau / (1 - tau)) ** 2) * max((1 - tau) ** 2, floor)
    target = (captured["x_t"] - captured["x0"]) / tau            # = m − x̂0
    torch.testing.assert_close(info["mixed"] - captured["x0"], target, rtol=1e-4, atol=1e-4)
    ref = info["is_weight"] * w * torch.nn.functional.mse_loss(fv, target)
    torch.testing.assert_close(loss_map["total_loss"].detach(), ref, rtol=1e-4, atol=1e-6)
    t = m._student_t_values()
    assert t[2] <= tau < t[1] - 0.01 + 1e-6


def test_fake_teacher_x0_fp32_matches_x0_head():
    m = make_method(student_rollout="ode")
    x_t = torch.randn(n_tok(), D)
    tt = torch.full((n_tok(),), 0.7)
    fx0, tx0 = m._fake_teacher_x0(x_t, tt, cond())
    with torch.no_grad():
        torch.testing.assert_close(fx0, m.fake_score(x_t, tt, condition=cond(), fwd_pred_type="x0", role="fake"))
        torch.testing.assert_close(tx0, m.teacher(x_t, tt, condition=cond(), fwd_pred_type="x0", role="teacher"))
    assert fx0.dtype == torch.float32


def test_vsd_xt_normalization():
    """Matches DP-DMD's train_sd35_dpdmd.py:146-152: grad = (p_real − p_fake)/mean|p_real|, p = x_t − x̂."""
    m = make_method(student_rollout="ode", dpdmd=True, vsd_norm="xt")
    torch.manual_seed(0)
    gen = torch.randn(n_tok(), D, requires_grad=True)
    x_t, real, fake = torch.randn(n_tok(), D), torch.randn(n_tok(), D), torch.randn(n_tok(), D)
    loss, _ = m._vsd_loss_norm(gen, real, fake, x_t)
    loss.backward()
    p_real, p_fake = x_t - real, x_t - fake
    grad = (p_real - p_fake) / p_real.abs().mean()
    ref = torch.autograd.grad(0.5 * torch.nn.functional.mse_loss(gen, (gen - grad).detach()), gen)[0]
    torch.testing.assert_close(gen.grad, ref)


def test_C_fake_uses_trajectory_endpoint_and_noise():
    m = make_method(student_rollout="ode", dmd_noise="trajectory", fake_input="final")
    m._sample_exit_step = lambda n: (_ for _ in ()).throw(AssertionError("fake_input=final must not sample an exit step"))
    captured = {}
    real_fwd = m.fake_score.forward

    def spy(x_t, t, condition=None, role="student", fwd_pred_type="v", **kw):
        captured.update(x_t=x_t.detach(), tau=float(t[0]))
        return real_fwd(x_t, t, condition=condition, role=role, fwd_pred_type=fwd_pred_type, **kw)

    m.fake_score.forward = spy
    noise = torch.randn(n_tok(), D)
    loss_map, out = m._rollout_fake_step(noise, cond())
    gen = out["gen_packed"]
    torch.testing.assert_close(gen, m.packed_generator_fn(m.net, noise, cond(), 4, "ode", T_LIST))
    tau = captured["tau"]
    torch.testing.assert_close(captured["x_t"], ((1 - tau) * gen + tau * noise).to(captured["x_t"].dtype))


def test_ode_init_random_is_independent_uniform():
    m = make_method(ode_init=True, student_update_freq=1, ode_init_points="random", rollout_exit="last")
    m._sample_exit_step = lambda n: (_ for _ in ()).throw(AssertionError("ode_init_points=random must not go through _sample_exit_step"))
    seen = set()
    for i in range(24):
        torch.manual_seed(i)
        reset(m)
        m.single_train_step(sample_batch(), iteration=3)
        seen |= {c["t"] for c in m.net.calls if c["role"] == "student" and c["grad"]}
    assert len(seen) == 4


# ── 13. fp32 master-copy optimizer, fp32 EMA accumulation, real PDD head layout ──

def test_adamw_master_matches_torch_adamw_fp32_and_moves_bf16():
    from fastgen.utils.adamw_master import AdamWMaster
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(64, 32))
    b = torch.nn.Parameter(a.detach().clone())
    oa = torch.optim.AdamW([a], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    ob = AdamWMaster([b], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    for i in range(5):
        g = torch.randn(64, 32)
        a.grad, b.grad = g.clone(), g.clone()
        oa.step(); ob.step()
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    # bf16: an lr-1e-5 step is below half an ulp, so torch AdamW's in-place updates are almost all
    # rounded away; the master-copy version accumulates them
    w0 = (torch.randn(4096) * 0.02).to(torch.bfloat16)
    pa, pb = torch.nn.Parameter(w0.clone()), torch.nn.Parameter(w0.clone())
    oa = torch.optim.AdamW([pa], lr=1e-5, weight_decay=0.0)
    ob = AdamWMaster([pb], lr=1e-5, weight_decay=0.0)
    for _ in range(50):
        g = torch.ones(4096, dtype=torch.bfloat16)
        pa.grad, pb.grad = g.clone(), g.clone()
        oa.step(); ob.step()
    moved_a = (pa != w0).float().mean().item()
    moved_b = (pb != w0).float().mean().item()
    assert moved_b > 0.9 and moved_b > moved_a + 0.3, (moved_a, moved_b)
    torch.testing.assert_close(ob.state[pb]["master"], w0.float() - 50 * 1e-5, rtol=0, atol=2e-6)


def test_ema_fp32_shadow_tracks_student():
    from fastgen.callbacks.ema import EMACallback
    torch.manual_seed(0)
    net = torch.nn.Linear(256, 256).to(torch.bfloat16)
    net.bias.requires_grad_(False)                               # frozen parameter: only copied at init
    ema = torch.nn.Linear(256, 256).to(torch.bfloat16)
    ema.load_state_dict(net.state_dict())
    ema.eval().requires_grad_(False)
    opt = torch.optim.SGD([net.weight], lr=0.0)
    model = types.SimpleNamespace(net=net, ema=ema, net_optimizer=opt)
    cb = EMACallback(type="constant", beta=0.995, fp32_shadow=True)
    cb.on_training_step_end(model, {}, {}, {}, iteration=0)      # start_iter=0: initialization
    ref = net.weight.detach().float().clone()
    for it in range(1, 201):
        with torch.no_grad():
            net.weight.add_(1e-4)                                 # student moves 1e-4 per step (far below bf16 ulp / (1−β))
        cb.on_training_step_end(model, {}, {}, {}, iteration=it)
        ref.lerp_(net.weight.detach().float(), 0.005)
    torch.testing.assert_close(ema.weight.float(), ref.to(torch.bfloat16).float(), rtol=0, atol=1e-6)
    # Control: a plain bf16 lerp(w, 1−β) stays put even after the student has moved ~10 ulp away
    # (dead zone ≈ ulp/(2(1−β)))
    e = torch.full((1000,), 0.5, dtype=torch.bfloat16)
    w = torch.full((1000,), 0.5 + 10 * 2 ** -9, dtype=torch.bfloat16)
    for _ in range(200):
        e.lerp_(w, 0.005)
    assert (e == 0.5).float().mean() > 0.9


def test_pdd_heads_real_layout():
    from fastgen.networks.kokoro.kokoro_network import KokoroNetwork
    torch.manual_seed(0)
    hidden, out, n = 48, 64, 8
    bagel = torch.nn.Module()
    bagel.llm2vae = torch.nn.Linear(hidden, out)
    bagel.pdd_heads = torch.nn.Linear(hidden, out * n)
    fake_self = types.SimpleNamespace(bagel=bagel, pdd_grid_steps=n)
    KokoroNetwork.init_pdd_heads_from_llm2vae(fake_self)
    h = torch.randn(10, hidden)
    u = bagel.pdd_heads(h).view(10, n, out)
    for j in range(n):
        torch.testing.assert_close(u[:, j], bagel.llm2vae(h))


def test_pdd_invalid():
    for bad in [dict(PDD_KW, student_update_freq=5), dict(PDD_KW, dpdmd=True), dict(PDD_KW, ode_init=True),
                dict(PDD_KW, pdd_grid_steps=30), dict(PDD_KW, pdd_l_min=3), dict(PDD_KW, pdd_solver="rk4"),
                dict(PDD_KW, pdd_l_max=4)]:
        with pytest.raises((ValueError, NotImplementedError)):
            make_method(**bad)


@pytest.mark.parametrize("arm", ["PDD", "PDD_euler", "ODEINIT_random"])
def test_single_train_step_new_arms(arm):
    kw = {"PDD": PDD_KW, "PDD_euler": dict(PDD_KW, pdd_solver="euler"),
          "ODEINIT_random": dict(ode_init=True, student_update_freq=1, ode_init_points="random")}[arm]
    m = make_method(**kw)
    m.config.sample_logging_iter = 5
    reset(m)
    loss_map, out = m.single_train_step(sample_batch(), iteration=5)
    assert torch.isfinite(loss_map["total_loss"])
    loss_map["total_loss"].backward()
    assert m.net.A.grad is not None and m.fake_score.A.grad is None
    assert "student_4nfe" in out["kokoro_x0_vis"]


# ── Cosine LR schedule: decay length = that optimizer's update count over iterations 1 .. MAX_ITER-1, reaching ~0 at the last update ──
def test_cosine_update_counts_and_schedule(monkeypatch):
    import importlib.util

    from fastgen.utils.lr_scheduler import LambdaWarmUpCosineScheduler

    monkeypatch.setenv("LR_SCHEDULE", "cosine")
    spec = importlib.util.spec_from_file_location(
        "cfg_kokoro_dmd2_test", os.path.join(os.path.dirname(__file__), "../../fastgen/configs/experiments/Kokoro/config_kokoro_dmd2.py"))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    # Same rule as DMD2.get_optimizers: the student updates when iteration % freq == 0, fake otherwise
    for max_iter, freq in [(1000, 5), (1000, 6), (1000, 2), (500, 1)]:
        n_net, n_fake = cfg.cosine_update_counts(max_iter, freq)
        its = range(1, max_iter)
        assert n_net == sum(1 for i in its if i % freq == 0)
        assert n_fake == (0 if freq == 1 else sum(1 for i in its if i % freq != 0))
    n_net, _ = cfg.cosine_update_counts(1000, 5)
    sch = LambdaWarmUpCosineScheduler(warm_up_steps=10, lr_min=0.0, lr_max=1.0, lr_start=1e-6, max_decay_steps=n_net)
    assert sch.schedule(10) == pytest.approx(1.0)                 # peak at the end of warmup
    assert sch.schedule(n_net - 1) < 1e-3                          # the n_net-th update (LambdaLR uses n_net-1) is near 0
    assert sch.schedule(n_net + 50) == pytest.approx(0.0)          # extra steps (e.g. misaligned resume) clamp to 0, no error


# ── TDM + DP-DMD first-step anchor (tdm_anchor_weight) ──────────────────────────

@pytest.mark.parametrize("k", [0, 1, 3])
def test_TDM_anchor_term(k):
    lam = 0.6
    m = make_method(student_rollout="ode", dmd_target="landing", tdm_anchor_weight=lam,
                    dpdmd_k_anchor=5, dpdmd_teacher_steps=30)
    force_exit(m, k)
    reset(m)
    torch.manual_seed(0)
    noise = torch.randn(n_tok(), D)
    loss_map, _ = m._rollout_student_step(noise, torch.randn(n_tok(), D), cond())
    t = m._student_t_values()
    # Anchor target, same formula as variant D: the teacher's mean velocity over 5 steps from t_0·ε
    grid = sorted({m._q(v) for v in shifted_grid(30, 3.0, t[0])}, reverse=True)
    x = noise * t[0]
    with torch.no_grad():
        for i in range(5):
            x = x + (grid[i + 1] - grid[i]) * m.teacher(x, torch.full((n_tok(),), grid[i]), condition=cond(),
                                                        role="teacher")
        v_target = (noise * t[0] - x) / (t[0] - grid[5])
        v_first = m.net(noise * t[0], torch.full((n_tok(),), t[0]), condition=cond(), role="student")
    torch.testing.assert_close(loss_map["anchor_loss"], torch.nn.functional.mse_loss(v_first, v_target))
    torch.testing.assert_close(loss_map["total_loss"], loss_map["vsd_loss"] + lam * loss_map["anchor_loss"])
    grad_calls = [c for c in m.net.calls if c["role"] == "student" and c["grad"]]
    assert len(grad_calls) == (1 if k == 0 else 2)          # k = 0 reuses the exit-step forward
    assert grad_calls[-1]["t"] == pytest.approx(t[0])
    loss_map["total_loss"].backward()
    assert m.net.A.grad is not None and m.fake_score.A.grad is None and m.teacher.A.grad is None


def test_TDM_anchor_off_is_unchanged():
    m = make_method(student_rollout="ode", dmd_target="landing")
    force_exit(m, 2)
    reset(m)
    loss_map, _ = m._rollout_student_step(torch.randn(n_tok(), D), torch.randn(n_tok(), D), cond())
    assert "anchor_loss" not in loss_map and loss_map["total_loss"] is loss_map["vsd_loss"]
    assert len([c for c in m.net.calls if c["role"] == "student" and c["grad"]]) == 1
    with pytest.raises(ValueError):
        make_method(student_rollout="ode", tdm_anchor_weight=0.6)          # the anchor is only valid with TDM (landing)


def test_TDM_anchor_decay_schedule():
    m = make_method(student_rollout="ode", dmd_target="landing", tdm_anchor_weight=0.6,
                    tdm_anchor_hold_iters=1000, tdm_anchor_final_ratio=0.5, tdm_anchor_end_iter=2500)
    assert m._anchor_weight(0) == pytest.approx(0.6) and m._anchor_weight(1000) == pytest.approx(0.6)
    assert m._anchor_weight(1750) == pytest.approx(0.45)          # cosine midpoint = (1 + 0.5) / 2 · 0.6
    assert m._anchor_weight(2500) == pytest.approx(0.3) and m._anchor_weight(3000) == pytest.approx(0.3)
    w = [m._anchor_weight(i) for i in range(1000, 2501, 50)]
    assert all(a >= b for a, b in zip(w, w[1:]))
    m2 = make_method(student_rollout="ode", dmd_target="landing", tdm_anchor_weight=0.6)    # no hold set = constant
    assert m2._anchor_weight(2400) == pytest.approx(0.6)
    # The weight actually used in the student step follows the iteration
    force_exit(m, 1)
    m._iteration = 2500
    loss_map, _ = m._rollout_student_step(torch.randn(n_tok(), D), torch.randn(n_tok(), D), cond())
    assert float(loss_map["anchor_weight"]) == pytest.approx(0.3)
    torch.testing.assert_close(loss_map["total_loss"], loss_map["vsd_loss"] + 0.3 * loss_map["anchor_loss"])
    with pytest.raises(ValueError):
        make_method(student_rollout="ode", dmd_target="landing", tdm_anchor_weight=0.6,
                    tdm_anchor_hold_iters=1000, tdm_anchor_final_ratio=1.5, tdm_anchor_end_iter=2500)
