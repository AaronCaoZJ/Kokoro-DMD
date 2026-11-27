# SPDX-License-Identifier: Apache-2.0
#
# C — DMDO (Hyper-Bagel, arXiv 2509.18824): B, but the VSD / fake-score noising reuses the noise that started the
# student's ODE trajectory instead of fresh noise (DMD_NOISE=trajectory); with FAKE_INPUT=final the fake score is fit
# on the full N-step rollout output.

from __future__ import annotations

import torch

from fastgen.methods.distribution_matching.kokoro_ode_common import _opt


class DMDOMixin:
    """Trajectory-noise switch of DMDO."""

    @property
    def _trajectory_noise(self) -> bool:
        return _opt(self.config, "dmd_noise", "fresh") == "trajectory"

    def _vsd_eps(self, noise: torch.Tensor, gen_data: torch.Tensor) -> torch.Tensor:
        """Noise for the VSD / fake-score forward process: the trajectory's starting noise under DMDO, fresh noise otherwise."""
        return noise.to(self.precision) if self._trajectory_noise else torch.randn_like(gen_data)
