# SPDX-License-Identifier: Apache-2.0
"""Average the gradients of *replicated* (non-FSDP) trainable params across ranks.

Kokoro/BAGEL keeps a few root LLM params out of FSDP sharding (``KokoroNetwork.fully_shard``:
embed_tokens / norm / norm_moe_gen / lm_head stay plain tensors so the per-rank KV-cache
build can call them directly).  FSDP only reduce-scatters the gradients of the params it
manages, so a trainable replicated param (``norm_moe_gen`` matches the ``moe_gen`` pattern)
would otherwise get a per-rank gradient and drift apart on each rank.  This callback runs at
``on_optimizer_step_begin`` -- register it BEFORE the grad-clip callbacks so clipping sees the
averaged gradient (and the global-norm all-reduce does not count it world_size times as
different values).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import world_size
import fastgen.utils.logging_utils as logger


class ReplicatedGradSyncCallback(Callback):
    def __init__(self, model_keys: Sequence[str] = ("net", "fake_score")) -> None:
        self.model_keys = list(model_keys)
        self._logged = False

    def on_optimizer_step_begin(self, model, iteration: int = 0) -> None:
        if world_size() <= 1:
            return
        names = []
        for key in self.model_keys:
            module = getattr(model, key, None)
            if module is None:
                continue
            for name, p in module.named_parameters():
                if p.requires_grad and p.grad is not None and not isinstance(p.grad, DTensor):
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size())
                    names.append(f"{key}.{name}")
        if names and not self._logged:
            logger.info(f"ReplicatedGradSyncCallback: averaging {len(names)} replicated grads, e.g. {names[:4]}")
            self._logged = True
