"""CPU checks for the DMD2 loss, f-divergence weighting and Decoupled-DMD (no checkpoints needed).

Mock UNets stand in for SD1.5; the checks cover grad mode of the teacher/fake forwards,
the f-divergence weights h(r), the CA timestep range and the real-UNet calls of Decoupled DMD.

Run from the try_decoupled-dmd root:  python tests/test_dmd_losses.py
"""
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import DDIMScheduler  # noqa: E402

# main.utils imports imageio only for its plotting helpers, which these tests do not use
# (stub it after diffusers, which probes imageio's import spec)
try:
    import imageio.v2  # noqa: F401
except ImportError:
    sys.modules["imageio"] = types.ModuleType("imageio")
    sys.modules["imageio.v2"] = types.ModuleType("imageio.v2")

from main.decoupled_guidance import SDDecoupledGuidance, sample_ca_timesteps  # noqa: E402
from main.f_divergence import FDivergenceWeighter  # noqa: E402
from main.sd_guidance import SDGuidance  # noqa: E402
from main.utils import NoOpContext  # noqa: E402


class MockUNet(nn.Module):
    """eps = w * x; records grad mode and batch size of every call."""

    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(w)))
        self.calls = []

    def forward(self, x, t, emb, added_cond_kwargs=None, classify_mode=False):
        self.calls.append({"grad": torch.is_grad_enabled(), "batch": x.shape[0], "classify": classify_mode})
        if classify_mode:
            return [x.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1).expand(-1, 1280, 2, 2) * self.w]
        return types.SimpleNamespace(sample=x * self.w)


def make_guidance(cls, use_f_divergence=False, divergence_type="JS", real_guidance_scale=3.0):
    g = cls.__new__(cls)
    nn.Module.__init__(g)
    g.fake_unet, g.real_unet = MockUNet(0.9), MockUNet(1.1)
    g.cls_pred_branch = nn.Conv2d(1280, 1, kernel_size=2)
    g.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    g.register_buffer("alphas_cumprod", g.scheduler.alphas_cumprod)
    g.num_train_timesteps, g.min_step, g.max_step = 1000, 20, 980
    g.real_guidance_scale, g.fake_guidance_scale = real_guidance_scale, 1.0
    g.use_fp16 = g.sdxl = False
    g.cls_on_clean_image = g.diffusion_gan = True
    g.diffusion_gan_max_timestep = 1000
    g.network_context_manager = NoOpContext()
    g.use_f_divergence, g.divergence_type = use_f_divergence, divergence_type
    if use_f_divergence:
        g.f_div_weighter = FDivergenceWeighter(divergence_type, 1000)
    return g


def test_f_div_weights():
    t = torch.tensor([100, 100, 900, 900])
    logits = torch.tensor([0.0, 1.0, -1.0, 2.0])

    h, _ = FDivergenceWeighter("reverse-KL", 1000)(logits, t)
    assert torch.allclose(h, torch.ones(4), atol=1e-5)

    # JS without time normalisation: h ∝ r/(1+r) = sigmoid(logit), mean 1
    h, _ = FDivergenceWeighter("JS", 1000, bin_num=0)(logits, t)
    ref = torch.sigmoid(logits)
    assert torch.allclose(h, ref / ref.mean(), atol=1e-5)

    # forward-KL: ratio is clamped to [0.1, 20] before weighting (exp(10) would dominate)
    h, stats = FDivergenceWeighter("forward-KL", 1000, bin_num=0)(torch.tensor([10.0, 0.0, 0.0, 0.0]), t)
    ref = torch.tensor([20.0, 1.0, 1.0, 1.0])
    assert torch.allclose(h, ref / ref.mean(), atol=1e-4), h
    assert abs(stats["f_div_weight_max"] - 20 / 5.75) < 1e-3

    # per-noise-level normalisation removes the scale difference between bins
    h, _ = FDivergenceWeighter("forward-KL", 1000)(torch.log(torch.tensor([2.0, 2.0, 8.0, 8.0])), t)
    assert torch.allclose(h, torch.ones(4), atol=1e-4), h


def test_ca_timesteps():
    torch.manual_seed(0)
    t = torch.tensor([999, 749, 499, 249, 20, 5]).repeat(4000)
    ca = sample_ca_timesteps(t, 20)
    ok = t > 20
    assert ((ca[ok] >= 20) & (ca[ok] < t[ok])).all()
    assert (ca[~ok] == 20).all()
    for tv in (999, 249):
        sel = ca[t == tv]
        assert sel.min() == 20 and sel.max() == tv - 1


def test_dm_loss_no_grad_and_f_div():
    torch.manual_seed(0)
    latents = torch.randn(4, 4, 8, 8, requires_grad=True)
    emb, uncond = torch.randn(4, 77, 8), torch.zeros(4, 77, 8)

    g = make_guidance(SDGuidance)
    torch.manual_seed(1)
    loss_dict, log = g.compute_distribution_matching_loss(latents, emb, uncond, None, None)
    assert not any(c["grad"] for c in g.fake_unet.calls + g.real_unet.calls), "teacher/fake forwards must run under no_grad"
    loss_dict["loss_dm"].backward()
    assert latents.grad is not None and g.fake_unet.w.grad is None and g.real_unet.w.grad is None
    plain_grad = log["dmtrain_grad"]

    gf = make_guidance(SDGuidance, use_f_divergence=True, divergence_type="forward-KL")
    torch.manual_seed(1)
    loss_dict_f, log_f = gf.compute_distribution_matching_loss(latents.detach().requires_grad_(), emb, uncond, None, None)
    cls_calls = [c for c in gf.fake_unet.calls if c["classify"]]
    assert len(cls_calls) == 1 and not cls_calls[0]["grad"]
    assert {"f_div_weight_std", "f_div_weight_max", "f_div_ratio_mean"} <= set(loss_dict_f)
    # same noise/timesteps (same seed) -> weighted grad is a per-sample rescale of the plain one
    ratio = log_f["dmtrain_grad"].flatten(1).norm(dim=1) / plain_grad.flatten(1).norm(dim=1)
    assert torch.allclose(ratio.mean(), torch.tensor(1.0), atol=1e-3), ratio


def test_decoupled_real_calls():
    torch.manual_seed(0)
    g = make_guidance(SDDecoupledGuidance)
    latents = torch.randn(3, 4, 8, 8, requires_grad=True)
    emb, uncond = torch.randn(3, 77, 8), torch.zeros(3, 77, 8)
    t = torch.tensor([999, 499, 249])
    loss_dict, log = g.compute_decoupled_loss(latents, emb, uncond, None, None, timesteps=t)
    # CA: one CFG call (cond+uncond, 2B); DM: cond only (B)
    assert [c["batch"] for c in g.real_unet.calls] == [6, 3], g.real_unet.calls
    assert not any(c["grad"] for c in g.real_unet.calls + g.fake_unet.calls)
    loss_dict["loss_decoupled"].backward()
    assert latents.grad is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
