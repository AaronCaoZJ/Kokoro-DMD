"""f-divergence weighting for distribution matching (f-distill, Xu et al. 2025).

The generator gradient of D_f(p_real || p_fake) is the DMD gradient re-weighted
per sample by h(r) = f''(r) * r^2, where r = p_real / p_fake is the density ratio
at the same noisy sample x_t and noise level t as the score difference.
A discriminator trained with the non-saturating softplus losses has optimal
logit log r, so r = exp(logit).

Mirrors the reference implementation in NVlabs/FastGen
(fastgen/methods/distribution_matching/f_distill.py): clamp the ratio, normalise
it per noise-level bin, then normalise h by its mean over the global batch.
"""
import torch
import torch.distributed as dist

F_DIV_WEIGHTING = {
    "reverse-KL": lambda r: torch.ones_like(r),  # f(r) = -log r  -> plain DMD
    "forward-KL": lambda r: r,                   # f(r) = r log r
    "JS": lambda r: r / (1 + r),                 # Jensen-Shannon
}


def _all_reduce_sum_(x):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x)
    return x


class FDivergenceWeighter:
    def __init__(self, divergence_type, num_train_timesteps,
                 ratio_lower=0.1, ratio_upper=20.0, bin_num=10, ratio_ema_rate=0.0):
        if divergence_type not in F_DIV_WEIGHTING:
            raise ValueError(f"Unknown divergence_type: {divergence_type}")
        self.divergence_type = divergence_type
        self.weighting = F_DIV_WEIGHTING[divergence_type]
        self.num_train_timesteps = num_train_timesteps
        self.ratio_lower = ratio_lower
        self.ratio_upper = ratio_upper
        self.bin_num = bin_num  # 0 disables per-noise-level normalisation
        self.ratio_ema_rate = ratio_ema_rate
        self.bins = None  # running mean ratio per noise-level bin, created lazily on device

    @torch.no_grad()
    def __call__(self, logits, timesteps):
        """logits: [B] or [B, 1] discriminator logits; timesteps: [B] DDPM indices.

        Must be called on every rank in the same step (it all-reduces).
        Returns h of shape [B] (mean 1 over the global batch) and a stats dict.
        """
        ratio = torch.exp(logits.float().reshape(-1).clamp(-10.0, 10.0))
        ratio = ratio.clamp(self.ratio_lower, self.ratio_upper)
        raw_ratio_mean = ratio.mean()

        if self.bin_num > 0:
            bin_width = self.num_train_timesteps / self.bin_num
            idx = (timesteps.double() / bin_width).floor().long().clamp(0, self.bin_num - 1)
            cnt = _all_reduce_sum_(torch.bincount(idx, minlength=self.bin_num).float())
            ratio_sum = _all_reduce_sum_(torch.bincount(idx, weights=ratio, minlength=self.bin_num).float())
            if self.bins is None:
                self.bins = torch.ones(self.bin_num, device=ratio.device)
            valid = cnt > 0
            new_vals = ratio_sum / cnt.clamp_min(1.0)
            self.bins[valid] = self.bins[valid] * self.ratio_ema_rate + (1 - self.ratio_ema_rate) * new_vals[valid]
            ratio = ratio / (self.bins[idx] + 1e-6)

        h = self.weighting(ratio)
        h_sum = _all_reduce_sum_(h.sum().reshape(1))
        n = _all_reduce_sum_(torch.tensor([float(h.numel())], device=h.device))
        h = h / (h_sum / n + 1e-6)

        stats = {
            "f_div_weight_std": h.std().item() if h.numel() > 1 else 0.0,
            "f_div_weight_max": h.max().item(),
            "f_div_weight_min": h.min().item(),
            "f_div_ratio_mean": raw_ratio_mean.item(),
        }
        return h, stats
