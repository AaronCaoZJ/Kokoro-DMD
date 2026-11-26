# Reader for the 1024px Z-Image packs of the reference training pipeline (the 1024px packed-data
# pipeline), adapting each sample to the format of the FastGen inference branch.
#   pack_reader.py     snapshot of the reference Z-Image pack reader and distributed shard split
#   zimage_caption.py  snapshot of the reference caption assembly (only VENDOR_DIR / SHARE now come from env vars)
#   caption rules      not shipped; KOKORO2_CAPTION_DIR points at a directory holding `Kokoro2/`, the pinned
#                      Kokoro2 caption rules (byte-identical)
#   loader.py          build_kokoro_mainline_pack_dataloader_for_trainer (reads kokoro_ode/data_mainline_1024.yaml)
from .loader import build_kokoro_mainline_pack_dataloader_for_trainer

__all__ = ["build_kokoro_mainline_pack_dataloader_for_trainer"]
