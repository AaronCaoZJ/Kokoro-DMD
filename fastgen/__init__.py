# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Kokoro: our BAGEL env is torch 2.5.1, where FSDP2 lives only under torch.distributed._composable.fsdp.
# FastGen imports it from torch.distributed.fsdp (torch>=2.6), so alias the names there once, before any submodule import.
import torch.distributed.fsdp as _fsdp

if not hasattr(_fsdp, "fully_shard"):
    from torch.distributed._composable import fsdp as _fsdp2

    for _name in ("fully_shard", "MixedPrecisionPolicy", "CPUOffloadPolicy", "OffloadPolicy", "FSDPModule"):
        setattr(_fsdp, _name, getattr(_fsdp2, _name))
