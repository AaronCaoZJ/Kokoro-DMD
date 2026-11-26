# Verbatim/trimmed port of the data pipeline of the reference CFG-distillation post-training code,
# so the FastGen Kokoro-DMD2 inference branch builds prompts byte-identically to it.
from .dataset import (
    LATENT_CHANNEL,
    LATENT_PATCH_SIZE,
    VAE_DOWNSAMPLE,
    infer_image_shape,
    make_dataloader,
    make_t2i_dataset,
    make_i2i_dataset,
    patchify_vae_latent,
)
from .tokens import add_special_tokens, pil_img2rgb
from .transforms import ImageTransform

__all__ = [
    "LATENT_CHANNEL",
    "LATENT_PATCH_SIZE",
    "VAE_DOWNSAMPLE",
    "infer_image_shape",
    "make_dataloader",
    "make_t2i_dataset",
    "make_i2i_dataset",
    "patchify_vae_latent",
    "add_special_tokens",
    "pil_img2rgb",
    "ImageTransform",
]
