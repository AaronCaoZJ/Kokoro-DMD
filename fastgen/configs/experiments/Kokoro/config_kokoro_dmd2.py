# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Experiment config for Kokoro-Bagel DMD2 distillation (T2I + I2I joint).
#
# Goal: distill the cfg-distilled multi-step Bagel teacher into a 4-NFE
# single-forward student that has NO cfg knob at inference (teacher cfg is
# absorbed into the student's moe_gen; student/fake never call the
# cfg embedders during training).
#
# Usage (single-node 8-GPU FSDP):
#   torchrun --nproc_per_node=8 train.py \
#       --config=fastgen/configs/experiments/Kokoro/config_kokoro_dmd2.py
#
# Required environment variables:
#   BAGEL_CKPT_DIR        — directory containing llm_config.json + ema.safetensors
#   DATA_ROOT_DIR         — root dir for WDS shard subdirectories
#   FASTGEN_OUTPUT_ROOT   — root dir for training outputs / checkpoints
#
# Optional env vars for shard config (else default globs under DATA_ROOT_DIR):
#   T2I_WDS_SHARD_DIRS    — comma-separated T2I subdir names
#   I2I_WDS_SHARD_DIRS    — comma-separated I2I subdir names

import copy
import os

import attrs
from omegaconf import DictConfig

from fastgen.callbacks.grad_clip import GradClipCallback
from fastgen.callbacks.replicated_grad_sync import ReplicatedGradSyncCallback
from fastgen.configs.callbacks import (
    EMA_CALLBACK,
    GPUStats_CALLBACK,
    ParamCount_CALLBACK,
    WANDB_CALLBACK,
)
from fastgen.configs.config import BaseConfig, BaseModelConfig
from fastgen.configs.discriminator import Discriminator_Kokoro_Config
from fastgen.configs.opt import BaseOptimizerConfig, BaseSchedulerConfig
from fastgen.datasets.kokoro_inference_wds import (
    build_kokoro_inference_dataloader_for_trainer,
)
from fastgen.methods.distribution_matching.kokoro_dmd2_ode import KokoroDMD2ODE
from fastgen.networks.kokoro.kokoro_network import KokoroNetwork
from fastgen.utils import LazyCall as L
from fastgen.utils.lr_scheduler import LambdaWarmUpCosineScheduler


# ── Paths ─────────────────────────────────────────────────────────────────────
_OUTPUT_ROOT = os.environ.get("FASTGEN_OUTPUT_ROOT", "outputs/kokoro-dmd2")
_DATA_ROOT = os.environ.get("DATA_ROOT_DIR", "/data")

# cfg-distill YAML — data-mix source of truth.  Loaded here (before I2I_PROB and
# the shard-spec block) because both reference _YAML.
_YAML = None
_CFG_DISTILL_YAML = os.environ.get("CFG_DISTILL_YAML", "")
if _CFG_DISTILL_YAML and os.path.isfile(_CFG_DISTILL_YAML):
    from omegaconf import OmegaConf
    _YAML = OmegaConf.to_container(OmegaConf.load(_CFG_DISTILL_YAML), resolve=True)

# 1024px Z-Image data pack: a non-empty DATA_YAML selects the kokoro_mainline_pack data adapter
# (e.g. kokoro_ode/data_mainline_1024.yaml); CFG_DISTILL_YAML / *_WDS_SHARD_DIRS are then ignored.
#   A non-empty I2I_PROB overrides the yaml's i2i_prob; a non-empty DATA_SEED overrides the yaml's read-order seed
#   (FastGen does not store the data-reading position, so restarting with the same seed rereads the same shards from
#   the start; pass a different seed when restarting).
DATA_YAML = os.environ.get("DATA_YAML", "").strip()
if DATA_YAML and not os.path.isfile(DATA_YAML):
    raise FileNotFoundError(f"DATA_YAML={DATA_YAML} does not exist")
_I2I_PROB_ENV = os.environ.get("I2I_PROB", "").strip()
_DATA_SEED_ENV = os.environ.get("DATA_SEED", "").strip()

# ── Training hyperparams (env-overridable) ────────────────────────────────────
SAVE_CKPT_ITER = int(os.environ.get("SAVE_CKPT_ITER", "500"))
STUDENT_UPDATE_FREQ = int(os.environ.get("STUDENT_UPDATE_FREQ", "5"))
STUDENT_SAMPLE_STEPS = int(os.environ.get("STUDENT_SAMPLE_STEPS", "4"))
STUDENT_FIRST_STEP_WEIGHT = float(os.environ.get("STUDENT_FIRST_STEP_WEIGHT", "1.0"))
SAMPLE_LOGGING_ITER = int(os.environ.get("SAMPLE_LOGGING_ITER", "50"))
I2I_PROB = float(os.environ.get("I2I_PROB") or (_YAML or {}).get("i2i_prob", 0.6))   # default matches the cfg-distill yaml

# Teacher CFG scales (only the teacher uses cfg_embed; student/fake do not).
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "4.0"))    # cfg_text
CFG_IMG_TEACHER = float(os.environ.get("CFG_IMG_TEACHER", "2.0"))  # i2i only

# i2i: teacher / fake_score prefill their own reference-image KV (the VAE prefix runs
# through the trainable moe_gen branch).  0 = share the student-built cache.
COND_KV_PER_ROLE = os.environ.get("COND_KV_PER_ROLE", "1").strip() not in ("0", "false", "False", "")

# Student schedule: "0.999,0.937,0.833,0.624,0.0" (empty = the default Wan table below).
_STUDENT_T_LIST_ENV = os.environ.get("STUDENT_T_LIST", "").strip()
STUDENT_T_LIST = (
    [float(v) for v in _STUDENT_T_LIST_ENV.split(",") if v.strip()]
    if _STUDENT_T_LIST_ENV else [0.999, 0.937, 0.833, 0.624, 0.0]
)

# ── ODE-rollout method switches (KokoroDMD2ODE) ───────────────────────────────
# Defaults = arm A (baseline).  B: STUDENT_ROLLOUT=ode; C: + DMD_NOISE=trajectory;
# D: + DPDMD=1; TDM: + DMD_TARGET=landing; ODE-init: ODE_INIT=1 (+ STUDENT_UPDATE_FREQ=1).
def _env_bool(name, default="0"):
    return os.environ.get(name, default).strip() not in ("0", "false", "False", "")

STUDENT_ROLLOUT = os.environ.get("STUDENT_ROLLOUT", "real_noised")
ROLLOUT_EXIT = os.environ.get("ROLLOUT_EXIT", "uniform")
FAKE_INPUT = os.environ.get("FAKE_INPUT", "matched")
DMD_NOISE = os.environ.get("DMD_NOISE", "fresh")
DMD_TARGET = os.environ.get("DMD_TARGET", "clean")
VSD_NORM = os.environ.get("VSD_NORM", "x0")
# VSD noise-level distribution: shifted with this shift (default 5; the official DP-DMD setting is 3).
VSD_T_SHIFT = float(os.environ.get("VSD_T_SHIFT", "5.0"))
TDM_MIN_T = float(os.environ.get("TDM_MIN_T", "0.02"))
TDM_MAX_MARGIN = float(os.environ.get("TDM_MAX_MARGIN", "0.01"))
TDM_HUBER_C = float(os.environ.get("TDM_HUBER_C", "1e-3"))   # TDM README --use_huber; 0 = mse/w
TDM_FAKE_W_FLOOR = float(os.environ.get("TDM_FAKE_W_FLOOR", "1.0"))   # TDM fake weight floor f (1 = default, 0 = demo-equivalent)
TDM_ANCHOR_WEIGHT = float(os.environ.get("TDM_ANCHOR_WEIGHT", "0"))   # TDM + DP-DMD first-step anchor weight λ; 0 = off
# Anchor schedule: λ held until TDM_ANCHOR_HOLD_ITERS, then cosine down to λ·TDM_ANCHOR_FINAL_RATIO at MAX_ITER;
# hold < 0 (default) = constant λ
TDM_ANCHOR_HOLD_ITERS = int(os.environ.get("TDM_ANCHOR_HOLD_ITERS", "-1"))
TDM_ANCHOR_FINAL_RATIO = float(os.environ.get("TDM_ANCHOR_FINAL_RATIO", "1.0"))
DPDMD = _env_bool("DPDMD")
DPDMD_DIV_WEIGHT = float(os.environ.get("DPDMD_DIV_WEIGHT", "0.05"))
DPDMD_TEACHER_STEPS = int(os.environ.get("DPDMD_TEACHER_STEPS", "30"))
DPDMD_TEACHER_SHIFT = float(os.environ.get("DPDMD_TEACHER_SHIFT", "3.0"))
DPDMD_K_ANCHOR = int(os.environ.get("DPDMD_K_ANCHOR", "5"))
DPDMD_GRAD_STEPS = os.environ.get("DPDMD_GRAD_STEPS", "after_first")
ODE_INIT = _env_bool("ODE_INIT")
ODE_INIT_TEACHER_STEPS = int(os.environ.get("ODE_INIT_TEACHER_STEPS", "40"))
ODE_INIT_TEACHER_SHIFT = float(os.environ.get("ODE_INIT_TEACHER_SHIFT", "3.0"))
ODE_INIT_POINTS = os.environ.get("ODE_INIT_POINTS", "all")
# PDD (parallel decoding distillation, arXiv 2607.26004): PDD=1 (+ STUDENT_UPDATE_FREQ=1).
PDD = _env_bool("PDD")
PDD_SOLVER = os.environ.get("PDD_SOLVER", "midpoint")
# paper: N = 128 for midpoint, 256 for euler (empty = that default)
PDD_GRID_STEPS = int(os.environ.get("PDD_GRID_STEPS", "").strip() or (256 if PDD_SOLVER == "euler" else 128))
PDD_SHIFT = float(os.environ.get("PDD_SHIFT", "5.0"))
PDD_L_MIN = int(os.environ.get("PDD_L_MIN", "0"))      # 0 = N/8 (paper: NFE 2/4/8 available)
PDD_L_MAX = int(os.environ.get("PDD_L_MAX", "0"))      # 0 = N/2

# ── GAN (inference-branch discriminator-feature path) ─────────────────────────
# Generator GAN loss weight: 0 = pure VSD+DSM;
# >0 builds the discriminator and activates the GAN paths. Conservative
# default 1e-3 (base DMD2 default; EDM recipe uses 3e-3) to gently
# break the VSD convergence plateau. R1 starts at 0 (enable a small value only
# after the discriminator is confirmed to be updating).
GAN_LOSS_WEIGHT_GEN = float(os.environ.get("GAN_LOSS_WEIGHT_GEN", "1e-3"))
GAN_R1_REG_WEIGHT = float(os.environ.get("GAN_R1_REG_WEIGHT", "0.0"))
GAN_R1_REG_ALPHA = float(os.environ.get("GAN_R1_REG_ALPHA", "0.1"))

BAGEL_CKPT_DIR = os.environ.get(
    "BAGEL_CKPT_DIR",
    "",  # BAGEL-7B-MoT directory (llm config, tokenizer, ae)
)
# Teacher / student init weights: CFG-distilled checkpoint.  Both student
# and teacher start from this; the student then diverges on moe_gen
# during training.
BAGEL_CKPT_PATH = os.environ.get(
    "BAGEL_CKPT_PATH",
    "",  # CFG-distilled teacher ema.safetensors
)

# max_latent_size: 64 → 1024px max (BAGEL-7B-MoT default), 96 → 1536px
MAX_LATENT_SIZE = int(os.environ.get("MAX_LATENT_SIZE", "128"))

# Optional: distinct student init (empty → student shares BAGEL_CKPT_PATH).
PRETRAINED_STUDENT_NET_PATH = os.environ.get("PRETRAINED_STUDENT_NET_PATH", "")

# ── Optimizer / LR-schedule hyperparams (env-overridable) ─────────────────────
NET_LR = float(os.environ.get("NET_LR", "1e-5"))
# fp32 master weights + moments in the optimizer (pure-bf16 params otherwise lose most Adam steps
# to rounding: half a bf16 ulp at |w|~0.02 is ~6e-5 > a 1e-5 step).  0 = plain torch AdamW.
OPTIM_FP32_MASTER = os.environ.get("OPTIM_FP32_MASTER", "1").strip() not in ("0", "false", "False", "")
NET_BETAS = tuple(float(v) for v in os.environ.get("NET_BETAS", "0.9,0.999").split(","))
FAKE_SCORE_BETAS = tuple(float(v) for v in os.environ.get("FAKE_SCORE_BETAS", "0.9,0.999").split(","))
FAKE_SCORE_LR = float(os.environ.get("FAKE_SCORE_LR", "2e-5"))
DISC_LR = float(os.environ.get("DISC_LR", "1e-5"))
NET_WARMUP_STEPS = int(os.environ.get("NET_WARMUP_STEPS", "10"))
FAKE_SCORE_WARMUP_STEPS = int(os.environ.get("FAKE_SCORE_WARMUP_STEPS", "50"))
DISC_WARMUP_STEPS = int(os.environ.get("DISC_WARMUP_STEPS", "50"))
# Learning-rate curve after the warm-up, for the student and the fake score: constant (default) or cosine to 0
# (as in CFG distillation).  Each scheduler steps only when its optimizer does, so the
# cosine length is that optimizer's number of updates in iterations 1 .. MAX_ITER-1 (the trainer loop).
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "constant").strip()


def cosine_update_counts(max_iter: int, student_update_freq: int) -> tuple[int, int]:
    """(student updates, fake-score updates) over the trainer's iterations 1 .. max_iter-1: the student updates
    when iteration % freq == 0 (DMD2.get_optimizers), the fake score on the other iterations; freq 1 = student only."""
    n_iter = max_iter - 1
    if student_update_freq <= 1:
        return n_iter, 0
    n_net = n_iter // student_update_freq
    return n_net, n_iter - n_net

assert LR_SCHEDULE in ("constant", "cosine"), f"LR_SCHEDULE must be constant | cosine, got {LR_SCHEDULE!r}"
MAX_ITER = int(os.environ.get("MAX_ITER", "50000"))
LOGGING_ITER = int(os.environ.get("LOGGING_ITER", "50"))
VALIDATION_ITER = int(os.environ.get("VALIDATION_ITER", "1000"))
BATCH_SIZE_GLOBAL = int(os.environ.get("BATCH_SIZE_GLOBAL", "32"))
SEED = int(os.environ.get("SEED", "0"))
CKPT_SUBDIR = os.environ.get("CKPT_SUBDIR", "checkpoints")
# Also emit a consolidated backend-loadable safetensors at every save
# (<ckpt>/exported/<iter>/student.safetensors[, ema.safetensors], step.txt).
# rank-0-only, best-effort. NOTE: each file is ~28GB, written every SAVE_CKPT_ITER
# steps — keep SAVE_CKPT_ITER sane (default 500) or set EXPORT_SAFETENSORS=0.
EXPORT_SAFETENSORS = os.environ.get("EXPORT_SAFETENSORS", "1") not in ("0", "false", "False", "")
# Maintain an EMA (exponential moving average) copy of the student net during
# training. Saved as the `ema` model key (-> <ckpt>.ema_model/ and, when export
# is on, exported/<iter>/ema.safetensors). EMA cannot be reconstructed after the
# fact — it only accumulates while a run with this enabled is training.
USE_EMA = os.environ.get("USE_EMA", "1") not in ("0", "false", "False", "")

# ── EMA local override ────────────────────────────────────────────────────────
# Env: EMA_RAMPUP_RATIO=<float>  → override the shared EMA_CALLBACK's rampup
# ratio FOR THIS config only, without mutating the module-level EMA_CALLBACK
# (which other method configs may import). We rebuild a fresh dict + fresh
# LazyCall here, so the global singleton in fastgen/configs/callbacks.py is
# untouched. Unset = the default rampup_ratio=0.05.
_EMA_RAMPUP_RATIO_OVERRIDE = os.environ.get("EMA_RAMPUP_RATIO", "").strip()
# EMA_FP32_SHADOW (default 1): fp32 accumulators for the trainable params (see EMACallback.fp32_shadow);
# 0 = bf16 lerp (dead band ~ulp/(2(1-beta)), so the EMA barely follows the student).
EMA_FP32_SHADOW = os.environ.get("EMA_FP32_SHADOW", "1").strip() not in ("0", "false", "False", "")
if USE_EMA and (_EMA_RAMPUP_RATIO_OVERRIDE or EMA_FP32_SHADOW):
    from fastgen.callbacks.ema import EMACallback
    _LOCAL_EMA_CALLBACK = dict(
        ema=L(EMACallback)(
            type="halflife",
            beta=0.9999,
            gamma=16.97,
            ema_halflife_kimg=500,
            ema_rampup_ratio=float(_EMA_RAMPUP_RATIO_OVERRIDE) if _EMA_RAMPUP_RATIO_OVERRIDE else 0.05,
            start_iter=0,
            fp32_shadow=EMA_FP32_SHADOW,
        ),
    )
else:
    _LOCAL_EMA_CALLBACK = EMA_CALLBACK if USE_EMA else {}
# FSDP meta-init: non-rank0 ranks build nets on the meta device (zero CPU alloc),
# then weights are broadcast from rank0. Cuts construction host-RAM from N*model to
# 1*model. REQUIRED with EMA on 8 GPUs: EMA setup does net.state_dict() copies that,
# replicated across all ranks, spike host RAM past the mem_guard limit and the
# job gets killed. With meta-init only rank0 pays that cost. Default on.
FSDP_META_INIT = os.environ.get("FSDP_META_INIT", "1") not in ("0", "false", "False", "")
PRECISION = os.environ.get("PRECISION", "bfloat16")
PRECISION_AMP = os.environ.get("PRECISION_AMP", "bfloat16")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "fastgen")
WANDB_GROUP = os.environ.get("WANDB_GROUP", "kokoro-dmd2")
WANDB_NAME = os.environ.get("WANDB_NAME", "bagel-dmd2-t2i-i2i")
WANDB_MODE = os.environ.get("WANDB_MODE", "online")


# ── Shard specs ───────────────────────────────────────────────────────────────
# Data-mix precedence: explicit *_WDS_SHARD_DIRS env  >  $CFG_DISTILL_YAML  >
# glob-all.  The cfg-distill YAML carries per-source path lists / weight /
# prompt_builder / decoder — exactly the structured form the loader consumes;
# the flat env route cannot express any of that.
def _specs_from_yaml(entries, is_i2i):
    specs = []
    for e in entries:
        s = {"path": e["path"], "weight": float(e.get("weight", 1.0))}
        if e.get("prompt_builder"):
            s["prompt_builder"] = e["prompt_builder"]
        if is_i2i and e.get("decoder"):
            s["decoder"] = e["decoder"]
        specs.append(s)
    return specs

# T2I
_T2I_SUBDIRS_ENV = os.environ.get("T2I_WDS_SHARD_DIRS", "") or os.environ.get(
    "WDS_SHARD_DIRS", ""
)
if _T2I_SUBDIRS_ENV:
    T2I_SHARDS = [
        {"path": os.path.join(_DATA_ROOT, d.strip(), "*.tar"),
         "weight": 1.0, "prompt_builder": "auto"}
        for d in _T2I_SUBDIRS_ENV.split(",") if d.strip()
    ]
elif _YAML and _YAML.get("t2i_shards"):
    T2I_SHARDS = _specs_from_yaml(_YAML["t2i_shards"], is_i2i=False)
else:
    T2I_SHARDS = [{"path": os.path.join(_DATA_ROOT, "*", "*.tar"),
                   "weight": 1.0, "prompt_builder": "auto"}]

# I2I
_I2I_SUBDIRS_ENV = os.environ.get("I2I_WDS_SHARD_DIRS", "")
if _I2I_SUBDIRS_ENV:
    I2I_SHARDS = [
        {"path": os.path.join(_DATA_ROOT, d.strip(), "*.tar"),
         "weight": 1.0, "decoder": "auto"}
        for d in _I2I_SUBDIRS_ENV.split(",") if d.strip()
    ]
elif _YAML and _YAML.get("i2i_shards"):
    I2I_SHARDS = _specs_from_yaml(_YAML["i2i_shards"], is_i2i=True)
else:
    I2I_SHARDS = []



# ── Network config (KokoroNetwork wrapping Bagel) ─────────────────────────────
KokoroNetwork_Config: DictConfig = L(KokoroNetwork)(
    bagel_ckpt_dir=BAGEL_CKPT_DIR,
    schedule_type="rf",
    net_pred_type="v",
    latent_patch_size=2,
    max_latent_size=MAX_LATENT_SIZE,
    visual_gen=True,
    visual_und=True,    # T2I+I2I joint training needs the und (VIT) branch loaded
    freeze_und=True,    # ... but kept frozen — DMD targets the gen branch
    timestep_shift=0.8,
    cfg_text_max=5.0,   # must match the cfg-distill checkpoint's training range
    cfg_img_max=2.5,
    pdd_grid_steps=PDD_GRID_STEPS if PDD else 0,   # PDD: N parallel copies of llm2vae
)


# ── DMD2 ModelConfig for Kokoro ───────────────────────────────────────────────
@attrs.define(slots=False)
class KokoroDMD2ModelConfig(BaseModelConfig):
    # Fake score optimizer (same arch as student).
    fake_score_optimizer: DictConfig = attrs.field(
        factory=lambda: copy.deepcopy(BaseOptimizerConfig)
    )
    fake_score_scheduler: DictConfig = attrs.field(
        factory=lambda: copy.deepcopy(BaseSchedulerConfig)
    )

    # Discriminator config objects.
    # gan_loss_weight_gen=0 keeps them unused at training time.
    discriminator: DictConfig = attrs.field(
        factory=lambda: copy.deepcopy(Discriminator_Kokoro_Config)
    )
    discriminator_optimizer: DictConfig = attrs.field(
        factory=lambda: copy.deepcopy(BaseOptimizerConfig)
    )
    discriminator_scheduler: DictConfig = attrs.field(
        factory=lambda: copy.deepcopy(BaseSchedulerConfig)
    )
    gan_loss_weight_gen: float = 0.0
    gan_use_same_t_noise: bool = True
    gan_r1_reg_weight: float = 0.0
    gan_r1_reg_alpha: float = 0.1

    # student vs fake_score update ratio (student step every N iters).
    student_update_freq: int = 5
    # 4-NFE preview interval (mirrors callbacks.wandb.sample_logging_iter). Must be a declared
    # field: Hydra composes the config via attrs.asdict, which drops undeclared attributes, and
    # the model would then silently fall back to logging_iter=50.
    sample_logging_iter: int = 50

    # Student inference NFE & schedule.
    student_sample_steps: int = 4
    student_first_step_weight: float = 1.0
    student_sample_type: str = "ode"   # Wan-style: deterministic to reduce train/test gap

    # Teacher cfg scales (only used when role="teacher" inside KokoroNetwork.forward).
    guidance_scale: float = 4.0       # cfg_text
    cfg_img_teacher: float = 2.0      # i2i cfg_img (apply_cfg_img_embed=True for i2i)

    fake_score_pred_type: str = "v"
    latent_patch_size: int = 2

    # Trainable scope: only moe_gen.  cfg_text_embedder /
    # cfg_img_embedder stay frozen (student/fake never call them, so they'd
    # receive zero gradient anyway).  Whole-net / text-path escalation is
    # blocked by the KV-cache runtime guard in
    # KokoroNetwork.apply_trainable_param_patterns.
    trainable_param_patterns: list = attrs.field(factory=lambda: ["moe_gen"])

    cond_kv_per_role: bool = True          # i2i: teacher/fake prefill their own reference-image KV

    # ODE-rollout switches (KokoroDMD2ODE).  Defaults reproduce arm A exactly.
    student_rollout: str = "real_noised"   # real_noised (A) | ode (B/C/D/TDM)
    rollout_exit: str = "uniform"          # exit step k ~ U{0..N-1} (rank0 broadcast) | last
    fake_input: str = "matched"            # matched (same distribution as the student step) | final
    dmd_noise: str = "fresh"               # fresh | trajectory (C: Hyper-Bagel DMDO)
    dmd_target: str = "clean"              # clean | landing (TDM)
    vsd_norm: str = "x0"                   # x0 = 1/mean|x - x_real| (DMD2) | xt = 1/mean|x_t - x_real| (DP-DMD)
    tdm_min_t: float = 0.02                # TDM lower bound of the per-step tau interval
    tdm_max_margin: float = 0.01           # TDM: tau < t_k - margin (demo: upt = t_g - 10)
    tdm_huber_c: float = 1e-3              # TDM generator pseudo-Huber c (README --use_huber); 0 = mse
    tdm_fake_w_floor: float = 1.0          # TDM fake v-MSE weight min(1,5/SNR)*max((1-tau)^2, f); 1 = default, 0 = demo-equivalent
    tdm_anchor_weight: float = 0.0         # TDM student step + λ·||v_1 − DP-DMD teacher anchor||² (first step from pure noise); 0 = off
    tdm_anchor_hold_iters: int = -1        # λ constant until this iteration, then cosine to λ·final_ratio at tdm_anchor_end_iter; <0 = constant
    tdm_anchor_final_ratio: float = 1.0
    tdm_anchor_end_iter: int = 0           # set to MAX_ITER by the experiment config
    dpdmd: bool = False
    dpdmd_div_weight: float = 0.05
    dpdmd_teacher_steps: int = 30
    dpdmd_teacher_shift: float = 3.0
    dpdmd_k_anchor: int = 5
    dpdmd_grad_steps: str = "after_first"  # after_first | last
    ode_init: bool = False
    ode_init_teacher_steps: int = 40
    ode_init_teacher_shift: float = 3.0
    ode_init_points: str = "all"           # all | random
    pdd: bool = False
    pdd_grid_steps: int = 128              # N (paper: 128 midpoint / 256 euler)
    pdd_shift: float = 5.0                 # grid shift (paper Qwen-Image: 5)
    pdd_solver: str = "midpoint"           # teacher mean velocity: midpoint (2 NFE) | euler (1 NFE, 2 intervals)
    pdd_l_min: int = 0                     # 0 = N/8
    pdd_l_max: int = 0                     # 0 = N/2


# ── Full experiment Config ────────────────────────────────────────────────────
@attrs.define(slots=False)
class KokoroDMD2Config(BaseConfig):
    model: KokoroDMD2ModelConfig = attrs.field(factory=KokoroDMD2ModelConfig)
    model_class: DictConfig = L(KokoroDMD2ODE)(config=None)


def create_config():
    config = KokoroDMD2Config()

    # ── Network ───────────────────────────────────────────────────────────────
    config.model.net = KokoroNetwork_Config

    # ── Pretrained weights ────────────────────────────────────────────────────
    config.model.pretrained_model_path = BAGEL_CKPT_PATH   # ema.safetensors
    config.model.pretrained_student_net_path = PRETRAINED_STUDENT_NET_PATH
    config.model.load_student_weights = True
    # EMA of the student net (requires the EMA callback registered below).
    config.model.use_ema = ["ema"] if USE_EMA else False

    # ── Precision ─────────────────────────────────────────────────────────────
    config.model.precision = PRECISION
    config.model.precision_amp = PRECISION_AMP
    # FSDP meta-init (see FSDP_META_INIT above): non-rank0 ranks build on the
    # meta device and receive the weights from rank 0.
    config.model.fsdp_meta_init = FSDP_META_INIT

    # ── Optimizers ────────────────────────────────────────────────────────────
    config.model.net_optimizer.lr = NET_LR
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_optimizer.betas = NET_BETAS
    config.model.fake_score_optimizer.lr = FAKE_SCORE_LR
    config.model.fake_score_optimizer.weight_decay = 0.0
    config.model.fake_score_optimizer.betas = FAKE_SCORE_BETAS
    if OPTIM_FP32_MASTER:
        config.model.net_optimizer.optim_type = "adamw_master"
        config.model.fake_score_optimizer.optim_type = "adamw_master"

    config.model.net_scheduler.warm_up_steps = [NET_WARMUP_STEPS]
    config.model.fake_score_scheduler.warm_up_steps = [FAKE_SCORE_WARMUP_STEPS]
    if LR_SCHEDULE == "cosine":
        # LR_SCHEDULE_ITERS: lay the cosine out over this many iterations while training still stops at MAX_ITER, so a
        # short run has the same per-step learning rate as the start of a longer run and can be compared with its
        # checkpoints at matching steps. Unset = MAX_ITER.
        n_net, n_fake = cosine_update_counts(int(os.environ.get("LR_SCHEDULE_ITERS") or MAX_ITER), STUDENT_UPDATE_FREQ)
        config.model.net_scheduler = L(LambdaWarmUpCosineScheduler)(
            warm_up_steps=NET_WARMUP_STEPS, lr_min=0.0, lr_max=1.0, lr_start=1.0e-6,
            max_decay_steps=max(n_net, NET_WARMUP_STEPS + 1))
        config.model.fake_score_scheduler = L(LambdaWarmUpCosineScheduler)(
            warm_up_steps=FAKE_SCORE_WARMUP_STEPS, lr_min=0.0, lr_max=1.0, lr_start=1.0e-6,
            max_decay_steps=max(n_fake, FAKE_SCORE_WARMUP_STEPS + 1))

    config.model.student_update_freq = STUDENT_UPDATE_FREQ

    # GAN via the inference-branch discriminator-feature path.
    # gan_loss_weight_gen>0 builds the discriminator (base DMD2.build_model) and
    # activates the generator/discriminator/R1 paths in KokoroDMD2.
    config.model.discriminator = Discriminator_Kokoro_Config
    config.model.discriminator.feature_indices = [13, 20, 27]
    config.model.gan_loss_weight_gen = GAN_LOSS_WEIGHT_GEN
    config.model.gan_use_same_t_noise = True
    config.model.gan_r1_reg_weight = GAN_R1_REG_WEIGHT
    config.model.gan_r1_reg_alpha = GAN_R1_REG_ALPHA
    config.model.discriminator_optimizer.lr = DISC_LR
    config.model.discriminator_optimizer.weight_decay = 0.0
    config.model.discriminator_scheduler.warm_up_steps = [DISC_WARMUP_STEPS]

    # ── Teacher CFG (teacher only) ────────────────────────────────────────────
    config.model.guidance_scale = GUIDANCE_SCALE
    config.model.cfg_img_teacher = CFG_IMG_TEACHER

    # ── Multi-step student (Wan-style) ────────────────────────────────────────
    config.model.student_sample_steps = STUDENT_SAMPLE_STEPS
    config.model.student_first_step_weight = STUDENT_FIRST_STEP_WEIGHT
    config.model.student_sample_type = "ode"
    # Wan T2V 4-step schedule.  Tune toward a higher-noise bias if the
    # intermediate steps stay blurry (e.g. [0.999, 0.95, 0.85, 0.65, 0.0]).
    # Env STUDENT_T_LIST overrides it (default = this table).
    config.model.sample_t_cfg.t_list = STUDENT_T_LIST

    config.model.cond_kv_per_role = COND_KV_PER_ROLE

    # ── ODE-rollout method switches (KokoroDMD2ODE) ───────────────────────────
    config.model.student_rollout = STUDENT_ROLLOUT
    config.model.rollout_exit = ROLLOUT_EXIT
    config.model.fake_input = FAKE_INPUT
    config.model.dmd_noise = DMD_NOISE
    config.model.dmd_target = DMD_TARGET
    config.model.vsd_norm = VSD_NORM
    config.model.tdm_min_t = TDM_MIN_T
    config.model.tdm_max_margin = TDM_MAX_MARGIN
    config.model.tdm_huber_c = TDM_HUBER_C
    config.model.tdm_fake_w_floor = TDM_FAKE_W_FLOOR
    config.model.tdm_anchor_weight = TDM_ANCHOR_WEIGHT
    config.model.tdm_anchor_hold_iters = TDM_ANCHOR_HOLD_ITERS
    config.model.tdm_anchor_final_ratio = TDM_ANCHOR_FINAL_RATIO
    config.model.tdm_anchor_end_iter = MAX_ITER
    config.model.dpdmd = DPDMD
    config.model.dpdmd_div_weight = DPDMD_DIV_WEIGHT
    config.model.dpdmd_teacher_steps = DPDMD_TEACHER_STEPS
    config.model.dpdmd_teacher_shift = DPDMD_TEACHER_SHIFT
    config.model.dpdmd_k_anchor = DPDMD_K_ANCHOR
    config.model.dpdmd_grad_steps = DPDMD_GRAD_STEPS
    config.model.ode_init = ODE_INIT
    config.model.ode_init_teacher_steps = ODE_INIT_TEACHER_STEPS
    config.model.ode_init_teacher_shift = ODE_INIT_TEACHER_SHIFT
    config.model.ode_init_points = ODE_INIT_POINTS
    config.model.pdd = PDD
    config.model.pdd_grid_steps = PDD_GRID_STEPS
    config.model.pdd_shift = PDD_SHIFT
    config.model.pdd_solver = PDD_SOLVER
    config.model.pdd_l_min = PDD_L_MIN
    config.model.pdd_l_max = PDD_L_MAX

    # VSD noise timestep distribution (Wan-style: shifted toward high t).
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = VSD_T_SHIFT
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # patch_dim = latent_patch_size² × vae_channels = 4 × 16 = 64.
    config.model.input_shape = [64]

    # ── Training ──────────────────────────────────────────────────────────────
    config.trainer.max_iter = MAX_ITER
    config.trainer.logging_iter = LOGGING_ITER
    config.trainer.validation_iter = VALIDATION_ITER
    config.trainer.save_ckpt_iter = SAVE_CKPT_ITER
    config.trainer.seed = SEED

    # Effective batch (default 32).  scripts.py reverse-derives grad_accum_rounds
    # from (batch_size_global / (dataloader.batch_size * world_size)).
    # 8-GPU FSDP → grad_accum_rounds=4.
    config.trainer.batch_size_global = BATCH_SIZE_GLOBAL

    # ── FSDP2 + activation checkpointing ──────────────────────────────────────
    # Shards all three KokoroNetworks (student / teacher / fake_score) per
    # KokoroNetwork.fully_shard.  Activation checkpointing is applied inside
    # fully_shard via apply_fsdp_checkpointing on each Qwen2 decoder layer.
    config.trainer.fsdp = True
    config.trainer.fsdp_min_num_params = 10_000_000

    config.trainer.checkpointer.save_dir = CKPT_SUBDIR
    config.trainer.checkpointer.export_safetensors = EXPORT_SAFETENSORS

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # Explicit GradClipCallback construction with posinf/neginf=±1.0 so any
    # leftover Inf gradient gets clamped into the [-1, 1] range before the
    # subsequent grad-norm clip (defence-in-depth).
    config.trainer.callbacks = DictConfig(
        {
            # Must stay first: average grads of the replicated (non-FSDP) trainable
            # params (norm_moe_gen) before the grad-clip callbacks run.
            "replicated_grad_sync": L(ReplicatedGradSyncCallback)(model_keys=["net", "fake_score"]),
            "grad_clip": L(GradClipCallback)(
                grad_norm=1.0,
                model_key="net",
                posinf=1.0,
                neginf=-1.0,
            ),
            "grad_clip_fake_score": L(GradClipCallback)(
                grad_norm=1.0,
                model_key="fake_score",
                posinf=1.0,
                neginf=-1.0,
            ),
            # Discriminator gradient clip (only relevant when GAN is on; the
            # callback is a no-op on steps where the discriminator has no grad).
            **(
                {
                    "grad_clip_discriminator": L(GradClipCallback)(
                        grad_norm=1.0,
                        model_key="discriminator",
                        posinf=1.0,
                        neginf=-1.0,
                    )
                }
                if GAN_LOSS_WEIGHT_GEN > 0
                else {}
            ),
            **_LOCAL_EMA_CALLBACK,  # see the "EMA local override" block above for env override semantics
            **GPUStats_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )
    config.trainer.callbacks.wandb.sample_logging_iter = SAMPLE_LOGGING_ITER
    # Mirror into the model config: KokoroDMD2._sample_logging_iter (the 4-NFE
    # preview gate in single_train_step) only has the model sub-config in hand,
    # not config.trainer, so it must read the value from its own namespace.
    config.model.sample_logging_iter = SAMPLE_LOGGING_ITER

    # ── Logging ───────────────────────────────────────────────────────────────
    config.log_config.project = WANDB_PROJECT
    config.log_config.group = WANDB_GROUP
    config.log_config.name = WANDB_NAME
    config.log_config.wandb_mode = WANDB_MODE

    # ── DataLoader: T2I + I2I mixed inference-style ──────────────────────────
    # MixedTaskLoader picks per step (rank 0 picks, broadcast to all ranks).
    if DATA_YAML:
        # 1024px Z-Image data pack; num_workers / shuffle_buffer / open_shards etc. come from the yaml,
        # and max_latent_size is the smaller of the yaml value and the network's MAX_LATENT_SIZE.
        from fastgen.datasets.kokoro_mainline_pack import build_kokoro_mainline_pack_dataloader_for_trainer

        config.dataloader_train = L(build_kokoro_mainline_pack_dataloader_for_trainer)(
            data_yaml=DATA_YAML,
            i2i_prob=float(_I2I_PROB_ENV) if _I2I_PROB_ENV else None,
            seed=int(_DATA_SEED_ENV) if _DATA_SEED_ENV else None,
            max_latent_size=MAX_LATENT_SIZE,
            batch_size=1,
            pin_memory=False,
        )
    else:
        # I2I_PROB defaults to 0.6 to match the cfg-distill yaml.
        config.dataloader_train = L(build_kokoro_inference_dataloader_for_trainer)(
            t2i_shards=T2I_SHARDS,
            i2i_shards=I2I_SHARDS,
            i2i_prob=I2I_PROB,
            batch_size=1,
            # 1 (not 2): host RAM = shuffle_buffer x num_workers x 2 loaders x ranks.
            # Both the t2i and i2i loaders each spawn num_workers, so this is a x2 on
            # the live decoded-sample footprint, which can exhaust host RAM. See
            # WDS_SHUFFLE_BUFFER note in kokoro_cfgdistill_data/dataset.py.
            num_workers=1,
            pin_memory=False,
        )

    return config
