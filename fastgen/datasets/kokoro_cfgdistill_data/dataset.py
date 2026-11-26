# Port of data/dataset.py of the reference CFG-distillation post-training code, with both T2I and I2I.
import os

import torch
from torch.utils.data import DataLoader
import webdataset as wds

from .prompt_builders_t2i import get_t2i_prompt_builder
from .prompt_builders_i2i import get_i2i_prompt_builder
from .sources_i2i import get_i2i_decoder
from .sample_utils import expand_shards, valid_latent


# ── Latent constants ──────────────────────────────────────────────────────────

LATENT_PATCH_SIZE = 2
LATENT_CHANNEL = 16
VAE_DOWNSAMPLE = 8  # latent spatial -> pixel: multiply by this

# Per-worker WDS shuffle buffer depth. This buffer holds *decoded* samples
# (the .decode() runs before .shuffle()), so for i2i story shards each slot is a
# fully PIL-decoded multi-frame sample (~200 MB). The live host-RAM footprint is
# WDS_SHUFFLE_BUFFER x num_workers x 2 loaders (t2i+i2i) x num_ranks, so large
# values (e.g. 128) can exhaust host RAM. Defaults to 16; override via env.
WDS_SHUFFLE_BUFFER = int(os.environ.get("WDS_SHUFFLE_BUFFER", "16"))


# ── Latent helpers ────────────────────────────────────────────────────────────


def patchify_vae_latent(latent, patch_size=LATENT_PATCH_SIZE):
    """[C, H_lat, W_lat] -> [num_tokens, patch_dim]"""
    c, h_lat, w_lat = latent.shape
    p = patch_size
    h, w = h_lat // p, w_lat // p
    latent = latent[:, : h * p, : w * p]
    latent = latent.reshape(c, h, p, w, p)
    latent = torch.einsum("chpwq->hwpqc", latent)
    return latent.reshape(h * w, p * p * c)


def infer_image_shape(latent, downsample=VAE_DOWNSAMPLE):
    """[C, H_lat, W_lat] -> (H_pixel, W_pixel)"""
    h_lat, w_lat = latent.shape[-2:]
    return h_lat * downsample, w_lat * downsample


# ── WebDataset decoders ───────────────────────────────────────────────────────


def _decode_t2i(sample, prompt_builder):
    """
    Tar layout per sample:  {id}.json  +  {id}.vae.pth
    WDS keys after .decode("pil"):
        "json"    -> dict (already parsed)
        "vae.pth" -> Tensor [1, C, H, W]
    """
    try:
        json_data = sample["json"]
        vae_latent = sample["vae.pth"].squeeze(0)  # [C, H, W]
        if not valid_latent(vae_latent):
            return None
        text = prompt_builder(json_data)
        if not text.strip():
            return None
        return {
            "text": text,
            "vae_latent": vae_latent,
            "source_image": None,
            "task_type": "t2i",
        }
    except Exception:
        return None


# ── Dataset / DataLoader factories ───────────────────────────────────────────


def make_t2i_dataset(shards, prompt_builder="auto", **_):
    prompt_builder_fn = get_t2i_prompt_builder(prompt_builder)
    return (
        wds.WebDataset(
            expand_shards(shards),
            resampled=True,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=wds.ignore_and_continue,
        )
        .decode("pil")
        .shuffle(WDS_SHUFFLE_BUFFER)
        .map(lambda sample: _decode_t2i(sample, prompt_builder=prompt_builder_fn))
        .select(lambda x: x is not None)
    )


def _decode_i2i(sample, decoder, prompt_builder=None):
    try:
        return decoder(sample, prompt_builder=prompt_builder) if prompt_builder is not None else decoder(sample)
    except Exception:
        return None


def make_i2i_dataset(shards, decoder="auto", prompt_builder=None, **_):
    decoder_fn = get_i2i_decoder(decoder)
    if prompt_builder is not None:
        # Validate the named builder exists (raises ValueError if not).
        get_i2i_prompt_builder(prompt_builder)
    return (
        wds.WebDataset(
            expand_shards(shards),
            resampled=True,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=wds.ignore_and_continue,
        )
        .decode("pil")
        .shuffle(WDS_SHUFFLE_BUFFER)
        .map(lambda s: _decode_i2i(s, decoder=decoder_fn, prompt_builder=prompt_builder))
        .select(lambda x: x is not None)
    )


def make_dataloader(dataset, num_workers=0, pin_memory=False):
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda samples: samples,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
