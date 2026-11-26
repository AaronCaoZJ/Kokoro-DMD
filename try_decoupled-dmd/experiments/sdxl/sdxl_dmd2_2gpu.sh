#!/bin/bash

# Work in the try_decoupled-dmd root regardless of where the script is called from
cd "$(dirname "$0")/../.."

# Set environment variables
export CHECKPOINT_PATH=${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the data/checkpoint root (see scripts/download_sdxl.sh)}
export WANDB_ENTITY=${WANDB_ENTITY:?set WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT:-sdxl_dmd2}
export PYTHONPATH=$(pwd):$PYTHONPATH

# HuggingFace model cache location
export HF_HOME=${HF_HOME:-$(pwd)/ckpt}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}

# Single node, 2 GPUs, FSDP
accelerate launch --config_file fsdp_configs/fsdp_1node_2x5090.yaml main/train_sd.py  \
    --generator_lr 5e-7  \
    --guidance_lr 5e-7 \
    --train_iters 100000000 \
    --output_path  $CHECKPOINT_PATH/sdxl_2gpu_test_lr5e-7_denoising4step_gan5e-3_guidance8  \
    --cache_dir $CHECKPOINT_PATH/cache \
    --log_path $CHECKPOINT_PATH/logs \
    --batch_size 1 \
    --grid_size 2 \
    --initialize_generator --log_iters 1000 \
    --resolution 1024 \
    --latent_resolution 128 \
    --seed 10 \
    --real_guidance_scale 8 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --wandb_iters 100 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_name "sdxl_2gpu_test_lr5e-7_denoising4step_gan5e-3_guidance8"  \
    --log_loss \
    --dfake_gen_update_ratio 5 \
    --fsdp \
    --sdxl \
    --use_fp16 \
    --max_step_percent 0.98 \
    --cls_on_clean_image \
    --gen_cls_loss \
    --gen_cls_loss_weight 5e-3 \
    --guidance_cls_loss_weight 1e-2 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --denoising \
    --num_denoising_step 4 \
    --denoising_timestep 1000 \
    --backward_simulation \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --real_image_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k_lmdb/
