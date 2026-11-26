Few-step distillation of the [BAGEL](https://github.com/ByteDance-Seed/Bagel) unified multimodal model (BAGEL-7B-MoT, text-to-image and image editing) with the DMD2 family of methods, built on [NVIDIA FastGen](https://github.com/NVlabs/FastGen). A separate folder, `try_decoupled-dmd/`, holds Decoupled-DMD and f-divergence experiments on Stable Diffusion.

# 1. Repository layout

| Path | Contents |
|---|---|
| `fastgen/networks/kokoro/` | BAGEL network (Qwen2 MoT LLM, SigLIP ViT, VAE) wrapped as a FastGen network |
| `fastgen/methods/distribution_matching/kokoro_dmd2.py` | `KokoroDMD2`: DMD2 (VSD + optional GAN) for BAGEL packed sequences |
| `fastgen/methods/distribution_matching/kokoro_dmd2_ode.py` | `KokoroDMD2ODE`: all variants in section 2.1, selected by switches |
| `fastgen/datasets/kokoro_*` | data adapters: CFG-distillation shards (default) and the 1024px packed-data pipeline (`DATA_YAML`) |
| `fastgen/{trainer.py,callbacks,configs,utils}` | minimal FastGen training framework (FSDP2, DCP checkpoints, EMA) |
| `kokoro_ode/` | launcher (`launch_arm.sh`), configs, two-stage pipeline, CPU unit tests, evaluation |
| `scripts/` | inference, batch evaluation, student export, host-RAM watchdog |
| `try_decoupled-dmd/` | DMD2 codebase (SD1.5 / SDXL) with Decoupled DMD and f-divergence weighting; see its README |

# 2. Methods we implemented

Every method below is our implementation.

- **What is upstream FastGen.** The base `DMD2Model` (VSD, fake-score training, GAN) and the trainer.
- **What is ours.** Everything else in section 2:
  - the BAGEL adaptation: packed sequences, MoT generation experts as the trainable part, and the CFG-distilled teacher embeddings;
  - all of the variants in 2.1;
  - the Stable Diffusion work in `try_decoupled-dmd/`.

## 2.1 BAGEL (`fastgen/`)

Select a variant with `ARM=` in `kokoro_ode/launch_arm.sh`. All variants distill to a 4-step student; other hyper-parameters come from `kokoro_ode/train_ode_dpdmd.yaml`. The notes record what we observed with BAGEL-7B-MoT; nothing here is a tuned benchmark result.

| ARM | Method | What changes relative to DMD2 | Reference | Notes from our runs |
|---|---|---|---|---|
| `A` | DMD2 | Student input is a noised real image | [DMD2](https://arxiv.org/abs/2405.14867) | Over-sharpens with `RECIPE=1`; `RECIPE=2` (shift-3 τ, β2 = 0.95, cosine LR to 0) largely fixes this by our metrics |
| `B` | DMD2 + ODE rollout | Student input comes from its own Euler rollout from noise to a random exit step (Self-Forcing style); one gradient-carrying step | Self Forcing (Huang et al., 2025) | Leaves a faint high-frequency grid; see section 2.3 for the inference-time fix |
| `C` | DMDO | `B` + re-noising with the trajectory's own noise; the fake score is fit on rollout end points | [Hyper-Bagel](https://arxiv.org/abs/2509.18824) | Training previews showed heavy contrast and saturation; stopped early |
| `D` | DP-DMD | The first student step is regressed to a multi-step teacher target; the DMD loss is applied to the later steps; x_t-based VSD normaliser; fake:student = 5:1 | [DP-DMD](https://arxiv.org/abs/2602.03139) | Sharpness closest to the teacher among the DMD variants; `RECIPE=2` reduces the grid artifact by about 20% (metric) |
| `TDM` | Trajectory Distribution Matching | Compares at each step's landing point and re-noises only inside that step's interval; importance weights; pseudo-Huber loss; fake:student = 1:1 | [TDM](https://arxiv.org/abs/2503.06674) | The fake-weight floor (`TDM_FAKE_W_FLOOR`) gave no measurable gain |
| `PDD` | Parallel Decoding Distillation | Multi-head output predicts several steps per evaluation; data-free; no fake score | [PDD](https://arxiv.org/abs/2607.26004) | Only run after ODE-init (see `ODEINIT_PDD`) |
| `ODEINIT` | Teacher-ODE initialisation | Online regression of every student step onto the teacher's ODE trajectory | CausVid-style ODE init | Correct composition but ghosting; used only as an initialisation |
| `ODEINIT_TDM` | ODE-init → TDM | Two stages via `kokoro_ode/run_odeinit_pipeline.sh` | — | Removes the ODE-init ghosting within a few hundred steps and has the least grid artifact, but is high-contrast. It drifts further from the teacher the longer it trains (closest around 500 steps); teacher CFG 3 and Huber c = 0.1 did not help. An optional first-step anchor (`TDM_ANCHOR_WEIGHT`) is implemented for this case |
| `ODEINIT_PDD` | ODE-init → PDD | Same as above, with PDD as the second stage | — | **Diverged** at lr 1e-5 with global batch 32: outputs degraded after ~150 steps and collapsed by ~180. Lower learning rates are untested |

`train_kokoro_dmd2.sh` with `train_config.yaml` runs plain `KokoroDMD2`: DMD2 + GAN on the CFG-distillation data.

## 2.2 Stable Diffusion (`try_decoupled-dmd/`)

- **Decoupled DMD** ([arXiv 2511.22677](https://arxiv.org/abs/2511.22677)). The CFG-augmentation term and the distribution-matching term are re-noised at separate timesteps.
- **f-divergence weighting.** The DMD gradient is re-weighted per sample (f-distill, JS / forward-KL / reverse-KL), with the density ratio taken from the diffusion-GAN discriminator.

Details and scripts are in `try_decoupled-dmd/README.md`.

## 2.3 Grid artifact at inference

The students' grid artifact is a residual, white-noise-like layer in the final latent. It can be removed at inference by re-noising the 4-step output to τ ≈ 0.1 and taking one more student step (5 NFE).

# 3. Setup

```bash
pip install -e .
PYTHONPATH=$PWD python -m pytest -q kokoro_ode/tests/test_ode_methods.py   # CPU, toy network in place of BAGEL
```

No paths are hard-coded:

- **Launcher-read YAML.** In `train_config.yaml` and `kokoro_ode/train_ode_dpdmd.yaml`, `${VAR}` is expanded from the environment, and variables already set in the environment take precedence.
- **OmegaConf-read YAML.** `cfg_distill.yaml` and `kokoro_ode/data_mainline_1024.yaml` use `${oc.env:VAR,''}`.

| Variable | Purpose |
|---|---|
| `BAGEL_MODEL_DIR` | BAGEL-7B-MoT directory (LLM/ViT configs, tokenizer, VAE) |
| `BAGEL_CKPT_DIR`, `BAGEL_CKPT_PATH` | model directory and initial teacher/student weights used by `config_kokoro_dmd2.py` |
| `TEACHER_CKPT`, `EVAL_TEACHER` | CFG-distilled teacher weights; the training YAMLs also initialise the student and fake score from `TEACHER_CKPT` |
| `DATA_ROOT_DIR`, `CFG_DISTILL_YAML` | CFG-distillation data (default data path) |
| `DATA_YAML`, `PACK_ROOT`, `HELDOUT_DIR` | 1024px packed data and the held-out evaluation set |
| `KOKORO2_CAPTION_DIR` | directory containing the `Kokoro2/` caption-assembly package (rules version `caption0819`, prompt mode `material`; checked at start-up). Needed only for the packed data; not shipped |
| `KOKORO_SHARE_DIR` | root of the caption side tables (`_sidetable_*`) and the Z-Image tokenizer used by the packed data; `*_SIDETABLE_DIR` / `CAPTION_TOKENIZER` override single directories (defaults in `CAPTION_ENV`, `zimage_caption.py`) |
| `KOKORO_TAG_FILES_DIR` | directory with the tag lists used by the CFG-distillation prompt builders (`R18_motion_tags.txt`, `Aplus-tag_deduplicated_with_counts.csv`) |
| `FASTGEN_OUTPUT_ROOT`, `RUNS_DIR` | training outputs and evaluation results |
| `PYTHON`, `EXTRA_PYTHONPATH` | interpreter and extra import paths for the launchers |
| `WANDB_ENTITY`, `WANDB_API_KEY` | set in the environment; never commit keys (`*.local.yaml` is git-ignored) |

The data YAMLs list the shards and pack directories of the original training mix; point `DATA_ROOT_DIR` / `PACK_ROOT` at your copies or edit the lists. For the packed data, every i2i pack directory must be listed in `I2I_MODE` (`pack_reader.py`), which marks it as `story` or `diff`.

# 4. Running

```bash
# one variant, single node
ARM=TDM GPUS=0,1,2,3 MASTER_PORT=29610 bash kokoro_ode/launch_arm.sh
# ODE-init, then TDM
PIPE=TDM ODE_INIT_ITERS=500 GPUS=0,1,2,3 MASTER_PORT=29610 bash kokoro_ode/run_odeinit_pipeline.sh
# DMD2 + GAN on the CFG-distillation data
bash train_kokoro_dmd2.sh
# sampling / side-by-side comparison of checkpoints
python scripts/inference/kokoro_inference.py --help
python scripts/inference/batch_eval.py --help
```

# 5. License

- **Repository.** Apache License 2.0 (`LICENSE`).
- **Upstream copyrights.** FastGen is copyright NVIDIA; the BAGEL model code is copyright ByteDance; the Qwen2 and SigLIP modelling code comes from HuggingFace transformers. See `licenses/README.md`; source files keep their original headers.
- **`try_decoupled-dmd/`.** This folder is derived from DMD2 and is licensed under **CC BY-NC-SA 4.0** (`try_decoupled-dmd/LICENSE.md`).
