#!/bin/bash

# DMD2 (4-step, GAN) with the DMD gradient re-weighted per sample by an f-divergence
# (f-distill). The density ratio comes from the diffusion-GAN discriminator, so
# --cls_on_clean_image and --diffusion_gan are required.

# Work in the try_decoupled-dmd root regardless of where the script is called from
cd "$(dirname "$0")/../.."

# Set environment variables
export CHECKPOINT_PATH=${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the data/checkpoint root (see scripts/download_sdv15.sh)}
export WANDB_ENTITY=${WANDB_ENTITY:?set WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT:-sdv15_dmd2}
export PYTHONPATH=$(pwd):$PYTHONPATH

# HuggingFace model cache location
export HF_HOME=${HF_HOME:-$(pwd)/ckpt}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}

accelerate launch --config_file fsdp_configs/fsdp_1node_2x5090_mix.yaml main/train_sd.py \
    --generator_lr 1e-5  \
    --guidance_lr 1e-5 \
    --train_iters 100000000 \
    --output_path  $CHECKPOINT_PATH/sdv15_dmd2_fdiv_js \
    --cache_dir $CHECKPOINT_PATH/cache \
    --log_path $CHECKPOINT_PATH/logs \
    --batch_size 4 \
    --grid_size 2 \
    --initialize_generator --log_iters 500 \
    --resolution 512 \
    --latent_resolution 64 \
    --seed 10 \
    --real_guidance_scale 3.0 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --real_image_path $CHECKPOINT_PATH/sd_vae_latents_laion_500k_lmdb \
    --wandb_iters 50 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_name "sdv15_dmd2_4step_cfg3.0_gan_fdiv_js"  \
    --wandb_project $WANDB_PROJECT \
    --use_fp16 \
    --log_loss \
    --dfake_gen_update_ratio 5 \
    --fsdp \
    --denoising \
    --num_denoising_step 4 \
    --denoising_timestep 1000 \
    --backward_simulation \
    --cls_on_clean_image \
    --gen_cls_loss \
    --gen_cls_loss_weight 5e-3 \
    --guidance_cls_loss_weight 1e-2 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --use_f_divergence \
    --divergence_type JS
