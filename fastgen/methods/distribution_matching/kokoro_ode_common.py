# SPDX-License-Identifier: Apache-2.0
#
# Shared pieces of the ODE-rollout methods (see kokoro_dmd2_ode.py for the method table).
#
# Notation (same as BAGEL / FastGen): x_t = (1−t)·x0 + t·ε, network output v = ε − x0,
#   x̂0 = x − t·v, ε̂ = x + (1−t)·v. One ODE step (= DDIM reusing the predicted noise = Euler): x_{t'} = x + (t' − t)·v.
#
# Numerical conventions:
#   * The rollout state x is accumulated in fp32 and cast to self.precision only at the network input
#     (multi-step accumulation in bf16 drifts).
#   * Every t used by the rollout is quantized to self.precision before the Euler update: the network's time
#     embedding consumes a bf16 t anyway, so the step size matches the t the network sees exactly
#     (in bf16, 0.999 → 1.0 and 0.624 → 0.625).
#   * Training rollouts, in-training previews (packed_generator_fn) and offline evaluation
#     (scripts/inference/batch_eval.py) share the module-level ode_rollout / ode_sample, so results from the
#     same noise are bit-identical.
#   * VSD noising / comparison / normalization reuse the parent class (bf16 τ, _vsd_loss).
#
# FSDP: every rank must run the same number of forwards in each single_train_step. The exit step k is drawn on
#   rank 0 and broadcast (_sample_exit_step / _broadcast_int).

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastgen.utils.distributed import is_rank0, synchronize, world_size


def _opt(config: Any, name: str, default: Any) -> Any:
    """Read a config field; if it is absent (older configs lack it) or None, return the default, i.e. the baseline behaviour."""
    val = getattr(config, name, default)
    return default if val is None else val


def shifted_grid(n_steps: int, shift: float, t_start: float, include: Sequence[float] = ()) -> List[float]:
    """Teacher ODE grid: an n_steps-step shifted schedule (s ∈ linspace(1,0), σ = shift·s / (1 + (shift−1)·s)),
    clipped at t_start (the student's start time), then merged with ``include`` (the student t_list; ODE-init needs
    the grid to pass exactly through these points).
    Returns a strictly decreasing list of floats ending at 0."""
    s = torch.linspace(1.0, 0.0, n_steps + 1, dtype=torch.float64)
    sig = shift * s / (1.0 + (shift - 1.0) * s)
    vals = {min(float(v), float(t_start)) for v in sig}
    vals |= {float(v) for v in include if float(v) <= float(t_start)}
    grid = sorted(vals, reverse=True)
    assert grid[0] == float(t_start) and grid[-1] == 0.0, grid
    return grid


def quantize_t(t: float, dtype: torch.dtype) -> float:
    """Quantize t to the network precision (the value the time embedding actually sees; in bf16, 0.999 → 1.0)."""
    return float(torch.tensor(float(t), dtype=torch.float64).to(dtype))


def net_velocity(net, x: torch.Tensor, t: float, condition: dict, dtype: torch.dtype, role: str) -> torch.Tensor:
    """One inference-branch forward returning the velocity v (in ``dtype``); x is the fp32 state and is cast to
    ``dtype`` before entering the network. Gradients follow the caller's context."""
    t_tok = torch.full((x.shape[0],), float(t), device=x.device, dtype=dtype)
    return net(x.to(dtype), t_tok, condition=condition, fwd_pred_type="v", role=role)


@torch.no_grad()
def ode_rollout(net, noise: torch.Tensor, t_vals: Sequence[float], k: int, condition: dict,
                dtype: torch.dtype, role: str = "student") -> torch.Tensor:
    """Start from x = t_0·ε and take k Euler steps along t_vals with the network's own velocity (= DDIM reusing the
    predicted noise); return the fp32 state at t_k.
    t_vals should already be quantized (quantize_t). Training rollouts, in-training previews and batch_eval all use this function."""
    x = noise.float() * t_vals[0]
    for i in range(k):
        v = net_velocity(net, x, t_vals[i], condition, dtype, role)
        x = x + (t_vals[i + 1] - t_vals[i]) * v.float()
    return x


@torch.no_grad()
def ode_sample(net, noise: torch.Tensor, t_vals: Sequence[float], condition: dict,
               dtype: torch.dtype, role: str = "student") -> torch.Tensor:
    """Full N-step ODE sampling; returns x̂0 (fp32) from the last step."""
    n_steps = len(t_vals) - 1
    x = ode_rollout(net, noise, t_vals, n_steps - 1, condition, dtype, role)
    v = net_velocity(net, x, t_vals[n_steps - 1], condition, dtype, role)
    return x - t_vals[n_steps - 1] * v.float()


class ODECommonMixin:
    """Time schedule, exit-step broadcast, rollout / teacher Euler, VSD comparison and previews shared by all arms."""

    # ── Student weights: accept exported safetensors (ODE-init output as the DMD starting point) ──

    def load_student_weights_and_ema(self) -> None:
        """When ``pretrained_student_net_path`` is ``exported/<iter>/student.safetensors`` (keys = BAGEL weights without
        the ``bagel.`` prefix), load it with load_bagel_weights: the parent's torch.load cannot read safetensors, and the
        different key prefix would be silently skipped by strict=False. Everything else goes to the parent."""
        path = self.config.pretrained_student_net_path
        if not (path and str(path).endswith(".safetensors") and self.config.load_student_weights):
            return super().load_student_weights_and_ema()

        from fastgen.networks.kokoro.kokoro_network import KokoroNetwork

        if (not self.config.fsdp_meta_init) or is_rank0():
            from safetensors import safe_open

            with safe_open(path, framework="pt") as f:
                file_keys = set(f.keys())
            # This branch does not copy the teacher weights first, so a key missing from the file would silently keep
            # its value from instantiation; require the full key set
            # (only PDD's pdd_heads may be missing; _setup_ema initializes them from llm2vae)
            want = set(self.net.bagel.state_dict().keys())
            missing = sorted(k for k in want - file_keys if not k.startswith("pdd_heads."))
            if missing:
                raise RuntimeError(
                    f"student weights {path} are missing {len(missing)} tensors (first 5: {missing[:5]}) -- "
                    "wrong key prefix, or not an exported student/ema.safetensors"
                )
            KokoroNetwork.load_bagel_weights(self.net.bagel, path)
            self._pdd_heads_loaded = "pdd_heads.weight" in file_keys
        synchronize()
        self._setup_ema()

    # ── Time schedule / helpers ───────────────────────────────────────────────

    def _q(self, t: float) -> float:
        return quantize_t(t, self.precision)

    def _student_t_values(self) -> List[float]:
        """Student time schedule (quantized), [t_0, …, t_{N−1}, 0], length N+1."""
        vals = [self._q(v) for v in self._get_student_t_list(device=torch.device("cpu")).tolist()]
        assert len(vals) == int(self.config.student_sample_steps) + 1, (
            f"t_list length {len(vals)} ≠ student_sample_steps+1 = {int(self.config.student_sample_steps) + 1}"
        )
        assert all(a > b for a, b in zip(vals[:-1], vals[1:])), f"t_list is not strictly decreasing after quantization: {vals}"
        return vals

    def _scalar(self, x: float) -> torch.Tensor:
        return torch.tensor(float(x), device=self.device)

    def _t_tok(self, t: float, n_tok: int) -> torch.Tensor:
        return torch.full((n_tok,), float(t), device=self.device, dtype=self.precision)

    def _net_v(self, net, x: torch.Tensor, t: float, condition: dict, role: str) -> torch.Tensor:
        return net_velocity(net, x, t, condition, self.precision, role)

    def _broadcast_int(self, value: int) -> int:
        if world_size() > 1:
            buf = torch.tensor(int(value), dtype=torch.long, device=self.device)
            dist.broadcast(buf, src=0)
            return int(buf.item())
        return int(value)

    def _sample_exit_step(self, n_steps: int) -> int:
        """Exit step k ∈ {0..N−1}: drawn on rank 0 and broadcast to all ranks (otherwise ranks run different numbers of
        forwards and FSDP collectives hang). rollout_exit=last always picks the last step; with uniform, step 0 is
        up-weighted by student_first_step_weight."""
        if _opt(self.config, "rollout_exit", "uniform") == "last":
            return n_steps - 1
        k = 0
        if is_rank0():
            probs = torch.ones(n_steps, dtype=torch.float64)
            probs[0] = max(1.0, float(_opt(self.config, "student_first_step_weight", 1.0)))
            k = int(torch.multinomial(probs / probs.sum(), 1).item())
        return self._broadcast_int(k)

    def _vsd_t(self, n_tok: int) -> torch.Tensor:
        """VSD noise level τ: same distribution as the parent (sample_t_cfg, shifted), quantized to self.precision and expanded to every token."""
        t_img = self._sample_t_per_image(1).to(self.precision)
        return t_img.expand(n_tok).contiguous()

    def _vsd_loss_norm(
        self, gen_data: torch.Tensor, teacher_x0: torch.Tensor, fake_x0: torch.Tensor, x_t_vsd: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """vsd_norm=x0: the parent's _vsd_loss (1/mean|x − x̂_real|, as in DMD2 and arm A).
        vsd_norm=xt: the original DP-DMD denominator 1/mean|x_t − x̂_real|; gradient direction, clipping and loss form match the parent."""
        if _opt(self.config, "vsd_norm", "x0") == "x0":
            return self._vsd_loss(gen_data, teacher_x0, fake_x0)
        with torch.no_grad():
            diff = (x_t_vsd.float() - teacher_x0.float()).abs().mean()
            weight_raw = 1.0 / (diff + 1e-6)
            weight = weight_raw.clamp(max=self._VSD_WEIGHT_MAX).to(gen_data.dtype)
            pseudo_target = gen_data - (fake_x0 - teacher_x0) * weight
        loss = (0.5 * F.mse_loss(gen_data.float(), pseudo_target.float(), reduction="mean")).to(gen_data.dtype)
        stats = {
            "vsd_weight": weight.detach(),
            "vsd_weight_raw_max": weight_raw.detach(),
            "vsd_weight_clipped": (weight_raw > self._VSD_WEIGHT_MAX).float().detach(),
            "student_teacher_l1": (gen_data - teacher_x0).abs().mean().detach(),
            "fake_teacher_l1": (fake_x0 - teacher_x0).abs().mean().detach(),
        }
        return loss, stats

    # ── ODE rollout ───────────────────────────────────────────────────────────

    def _ode_rollout_to(
        self, noise: torch.Tensor, t_vals: Sequence[float], k: int, condition: dict, net=None
    ) -> torch.Tensor:
        """no_grad: the student takes k Euler steps from t_0·ε to t_k (module-level ode_rollout). Same starting point as packed_generator_fn."""
        return ode_rollout(self.net if net is None else net, noise, t_vals, k, condition, self.precision)

    def _exit_forward(self, x_k: torch.Tensor, t_k: float, condition: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Exit-step forward (gradients follow the caller's context): returns (v, x̂0), with x̂0 in fp32."""
        v = self._net_v(self.net, x_k, t_k, condition, "student")
        return v, x_k - t_k * v.float()

    def _teacher_euler(self, x: torch.Tensor, grid: Sequence[float], n_steps: int, condition: dict,
                       record: Optional[Dict[int, float]] = None) -> tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """no_grad teacher Euler (cfg embedding, role=teacher): take n_steps steps along grid.
        record = {name: t}: save the state when the rollout reaches t (used by ODE-init)."""
        states: Dict[int, torch.Tensor] = {}
        want = {} if record is None else {t: name for name, t in record.items()}
        with torch.no_grad():
            for i in range(n_steps):
                if grid[i] in want:
                    states[want[grid[i]]] = x.clone()
                v = self._net_v(self.teacher, x, grid[i], condition, "teacher")
                x = x + (grid[i + 1] - grid[i]) * v.float()
        return x, states

    # ── VSD comparison ────────────────────────────────────────────────────────

    def _fake_teacher_x0(
        self, x_t_vsd: torch.Tensor, t_vsd: torch.Tensor, condition: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fake and teacher x0 predictions on x_τ (no_grad, same order as the parent). The networks output v and
        x0 = x_τ − τ·v is computed in fp32: inside the network that step is bf16 (ulp 0.004–0.008), and the rounding
        noise from subtracting two separately rounded x0s exceeds the true fake−teacher gap
        (especially with TDM's Huber c=1e-3)."""
        tau = t_vsd.float().unsqueeze(-1)
        with torch.no_grad():
            fake_v = self.fake_score(x_t_vsd, t_vsd, condition=condition, fwd_pred_type="v", role="fake")
            teacher_v = self.teacher(x_t_vsd, t_vsd, condition=condition, fwd_pred_type="v", role="teacher")
        x_t = x_t_vsd.float()
        return x_t - tau * fake_v.float(), x_t - tau * teacher_v.float()

    def _dmd_compare(
        self, gen_data: torch.Tensor, x_t_vsd: torch.Tensor, t_vsd: torch.Tensor, condition: dict
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Fake and teacher x0 predictions on x_τ (fp32); the VSD gradient is applied to gen_data."""
        fake_x0, teacher_x0 = self._fake_teacher_x0(x_t_vsd, t_vsd, condition)
        loss, stats = self._vsd_loss_norm(gen_data, teacher_x0, fake_x0, x_t_vsd)
        return loss, stats, fake_x0, teacher_x0

    # ── Preview ───────────────────────────────────────────────────────────────

    def _attach_4nfe_preview(self, outputs: dict, condition: dict, iteration: int) -> None:
        """Same gate as the parent's single_train_step: on iteration 1 and every sample_logging_iter student steps, run one full N-step inference."""
        every = self._sample_logging_iter()
        if not (iteration == 1 or (every > 0 and iteration % every == 0)):
            return
        with torch.no_grad():
            gen = outputs["gen_packed"]
            preview = self.packed_generator_fn(
                net=self.net_inference,
                noise_packed=torch.randn(gen.shape, device=gen.device, dtype=torch.float32),
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=getattr(self.config, "student_sample_type", "ode"),
                t_list=getattr(self.config.sample_t_cfg, "t_list", None),
                precision_amp=self.precision_amp_infer,
            )
        shape = condition["patchified_vae_latent_shapes"][0]
        outputs["kokoro_x0_vis"]["student_4nfe"] = preview[: shape[0] * shape[1]].detach()
