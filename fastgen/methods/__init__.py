# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Kokoro-DMD keeps only the DMD2 base model and the Kokoro/BAGEL distillation methods.

from fastgen.methods.model import FastGenModel as FastGenModel
from fastgen.methods.distribution_matching.dmd2 import DMD2Model as DMD2Model
from fastgen.methods.distribution_matching.kokoro_dmd2 import KokoroDMD2 as KokoroDMD2
