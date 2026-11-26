Decoupled-DMD and f-divergence DMD on Stable Diffusion, built on the [DMD2](https://github.com/tianweiy/DMD2) codebase (SD1.5 / SDXL, `diffusers` UNets, `accelerate` + FSDP). This folder is self-contained and independent of the BAGEL code in the parent repository.

> **License:** this folder keeps DMD2's **CC BY-NC-SA 4.0** license (`LICENSE.md`, non-commercial, share-alike), which differs from the Apache-2.0 license of the parent repository.

# 1. What we added on top of DMD2

Everything in this table is our implementation; the rest of the folder is upstream DMD2.

| Method | Code | Flags | Reference |
|---|---|---|---|
| **Decoupled DMD** | `main/decoupled_guidance.py` | `--use_decoupled_dmd` (needs `--real_guidance_scale > 1`) | *Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield*, [arXiv 2511.22677](https://arxiv.org/abs/2511.22677) |
| **f-divergence weighting** of the DMD gradient | `main/f_divergence.py`, `main/sd_guidance.py` | `--use_f_divergence --divergence_type {JS,forward-KL,reverse-KL}`; `--f_div_ratio_lower/upper`, `--f_div_bin_num`, `--f_div_ratio_ema` | *One-step Diffusion Models with f-Divergence Distribution Matching* (f-distill, Xu et al., 2025); weighting follows [NVlabs/FastGen](https://github.com/NVlabs/FastGen) `f_distill.py` |
| **Multi-step SD1.5 with backward simulation** | `main/sd_unified_model.py`, `main/train_sd.py` | `--denoising --num_denoising_step 4 --backward_simulation` on SD1.5 | DMD2 only supported this for SDXL |
| FSDP + bf16 mixed precision, 2-GPU configs | `main/train_sd.py`, `fsdp_configs/fsdp_1node_2x5090*.yaml` | `--fsdp --use_fp16` | — |
| Few-step / teacher sampling with CLIP score | `main/simple_inference*.py`, `scripts/simple_eval.sh`, `scripts/teacher_eval.sh` | — | — |
| CPU tests | `tests/test_dmd_losses.py` | `python tests/test_dmd_losses.py` | — |

## 1.1 Decoupled DMD

With a CFG teacher, the DMD update splits into two terms. The DM term is `x0_real_cond − x0_fake`, the plain distribution-matching direction. The CA term is `(w − 1)(x0_real_cond − x0_real_uncond)`, the CFG augmentation. Decoupled DMD re-noises the generated sample separately for each term, with independent noise:

- **CA term**: timestep `τ_CA ∈ [min_step, t)`, i.e. always cleaner than the generator's input timestep `t`. This matches Algorithm 1 of the paper, `τ_CA ~ U(t, 1)`; the paper writes time as 0 = noise, while here the DDPM index 999 is noise.
- **DM term**: timestep `τ_DM` over the full `[min_step, max_step]` range.
- **Fake score**: trained exactly as in DMD2, with the `--dfake_gen_update_ratio` two-timescale updates.

Notes on where this implementation deviates from the paper or from the DMD2 baseline:

- **Normalisation.** Each term is normalised DMD2-style by `mean |x0_real_cond − x|`, computed at its own timestep. The paper does not specify a normaliser.
- **Loss scale.** The loss is `mse(x, sg(x + Δ))` without DMD2's `0.5` factor. Its gradient is therefore 2× that of the DMD2 baseline, which halves the effective weight of the GAN term (`--gen_cls_loss_weight`) relative to it. Keep this in mind when comparing against `sdv15_dmd2_accelerate.sh`.
- **f-divergence.** `--use_f_divergence` is ignored on this path.

## 1.2 f-divergence weighting

The DMD2 generator gradient is re-weighted per sample by `h(r) = f''(r)·r²`, where `r = p_real / p_fake`:

| `--divergence_type` | `h(r)` | Effect |
|---|---|---|
| `reverse-KL` | `1` | plain DMD |
| `forward-KL` | `r` | |
| `JS` | `r / (1 + r)` | |

How `r` and `h` are computed:

- **Ratio estimate.** `r = exp(logit)` comes from the GAN discriminator, so `--cls_on_clean_image` is required.
- **Where it is evaluated.** With `--diffusion_gan`, the logit is evaluated on the same noisy sample `(x_t, t)` as the DMD score difference. Without it, the logit falls back to the clean sample.
- **Ratio normalisation.** `r` is clamped to `[0.1, 20]` and normalised by its running mean per noise-level bin (`--f_div_bin_num`; `0` disables this).
- **Weight normalisation.** `h` is normalised to mean 1 over the global batch (all-reduced across ranks).
- **Logging.** W&B logs `f_div_weight_{std,max,min}` and `f_div_ratio_mean`.

# 2. Usage

```bash
pip install -r requirements.txt && python setup.py develop
export CHECKPOINT_PATH=<data and checkpoint root>
bash scripts/download_sdv15.sh $CHECKPOINT_PATH      # LAION captions + SD1.5 VAE latents (SDXL: download_sdxl.sh)
export WANDB_ENTITY=<your wandb entity>
```

Training scripts (single node, 2 GPUs, FSDP):

| Script | Setting |
|---|---|
| `experiments/sdv1.5/sdv15_dmd2_accelerate.sh` | DMD2 baseline: 4-step, backward simulation, diffusion GAN, CFG 3.0 |
| `experiments/sdv1.5/sdv15_dmd2_fdiv.sh` | Same, plus f-divergence weighting (JS) |
| `experiments/sdv1.5/sdv15_decoupled_dmd.sh` | Decoupled DMD without GAN, CFG 7.5, full timestep range |
| `experiments/sdv1.5/sdv15_ddmd_gan.sh` | Decoupled DMD + diffusion GAN, CFG 1.75 |
| `experiments/sdxl/sdxl_dmd2_2gpu.sh` | SDXL DMD2, 4-step, on 2 GPUs |
| `experiments/sdxl/sdxl_decoupled_dmd.sh` | SDXL Decoupled DMD, multi-node (`CHECKPOINT_PATH WANDB_ENTITY WANDB_PROJECT FSDP_DIR RANK` as arguments) |

Sampling and checks:

```bash
CHECKPOINT=<output_path>/<run>/checkpoint_model_XXXXXX/pytorch_model.bin bash scripts/simple_eval.sh   # student, CLIP score
bash scripts/teacher_eval.sh                                                                          # SD1.5 teacher reference
python tests/test_dmd_losses.py                                                                       # CPU, no checkpoints
```

Changes that affect upstream DMD2 commands:

- **Renamed flag.** `--initialie_generator` is now `--initialize_generator`; all scripts in `experiments/` are updated.
- **Original instructions.** For ImageNet and the original SDXL recipes, see `experiments/*/README.md` and the [upstream README](https://github.com/tianweiy/DMD2).

# 3. Citation

```bibtex
@inproceedings{yin2024improved,
  title={Improved Distribution Matching Distillation for Fast Image Synthesis},
  author={Yin, Tianwei and Gharbi, Micha{\"e}l and Park, Taesung and Zhang, Richard and Shechtman, Eli and Durand, Fredo and Freeman, William T},
  booktitle={NeurIPS},
  year={2024}
}
```

Also cite the Decoupled DMD and f-distill papers above when using those parts.
