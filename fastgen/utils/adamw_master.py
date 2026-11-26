# SPDX-License-Identifier: Apache-2.0
"""AdamW with an fp32 master copy for low-precision (bf16) parameters.

Kokoro/BAGEL trains in pure bf16 (precision = precision_fsdp = bf16, no autocast).  Plain
torch AdamW then updates the bf16 params in place: with |w| ~ 0.02 the bf16 ulp is ~1.2e-4, so
any Adam step below half an ulp (~6e-5; lr 1e-5 gives steps of ~1e-5) rounds back to the old
value and most weights never move.

Here every param keeps, in the optimizer state, an fp32 ``master`` copy plus fp32 moments; the
update is done on the master and the param is overwritten with its bf16 rounding.  The state is
keyed by the model param (as in torch AdamW), so FSDP2 / DCP optimizer checkpoints keep working
(DTensor params get DTensor states with the same placements).  Math = torch.optim.AdamW
(decoupled weight decay, bias-corrected moments).  Resuming from a plain-AdamW checkpoint seeds
the master from the current param and upcasts the moments.
"""

from __future__ import annotations

import math

import torch


class AdamWMaster(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, **_ignored):
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _init_state(p: torch.Tensor, st: dict) -> None:
        if "master" not in st:
            st["master"] = p.detach().clone().float()
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in st:
                st[key] = torch.zeros_like(st["master"])
            elif st[key].dtype != torch.float32:
                st[key] = st[key].float()
        if "step" not in st:
            st["step"] = torch.zeros((), dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                self._init_state(p, st)
                st["step"] += 1
                step = float(st["step"])
                g = p.grad.float()
                w, m, v = st["master"], st["exp_avg"], st["exp_avg_sq"]
                if wd != 0:
                    w.mul_(1.0 - lr * wd)
                m.lerp_(g, 1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                bc1 = 1.0 - b1 ** step
                bc2 = 1.0 - b2 ** step
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                w.addcdiv_(m, denom, value=-lr / bc1)
                p.copy_(w)
        return loss
