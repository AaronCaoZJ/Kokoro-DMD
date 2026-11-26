"""
Standard multi-step sampling with the original SD1.5 teacher,
for comparison with the distilled model's outputs.
Supports several schedulers, including LCM for few-step comparisons.
"""
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, LCMScheduler
from PIL import Image
import torch
import argparse
import os


def save_images(images, prompts, output_dir):
    """Save the generated images to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (image, prompt) in enumerate(zip(images, prompts)):
        # filesystem-safe file name from the prompt
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
        safe_prompt = safe_prompt[:50]
        
        filename = f"{idx:04d}_{safe_prompt}.png"
        filepath = os.path.join(output_dir, filename)
        
        image.save(filepath)
        print(f"Saved: {filepath}")
    
    print(f"\nAll images saved to {output_dir}")


@torch.no_grad()
def generate_images_teacher(pipeline, prompts, args):
    """Generate images with the teacher pipeline."""
    all_images = []
    
    print(f"Sampling with {args.num_inference_steps} steps")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Scheduler: {args.scheduler}")
    
    # LCM with user-specified fixed timesteps
    if args.scheduler == "lcm" and args.lcm_timesteps:
        print(f"LCM fixed timesteps: {args.lcm_timesteps}")
    
    for idx, prompt in enumerate(prompts):
        print(f"Generating image {idx+1}/{len(prompts)}: {prompt}")
        
        # per-image seed, same as the distilled-model inference scripts
        generator = torch.Generator(device=args.device).manual_seed(args.seed + idx)
        
        # standard diffusers pipeline
        if args.scheduler == "lcm" and args.lcm_timesteps:
            # LCM with fixed timesteps (like DMD2)
            image = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.image_resolution,
                width=args.image_resolution,
                timesteps=args.lcm_timesteps,
            ).images[0]
        else:
            # Standard sampling
            image = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.image_resolution,
                width=args.image_resolution,
            ).images[0]
        
        all_images.append(image)
    
    return all_images


def main():
    parser = argparse.ArgumentParser(description="Teacher model inference for SD1.5 with standard sampling")
    parser.add_argument("--output_dir", type=str, default="./teacher_outputs",
                        help="Output directory")
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=[
                            "a beautiful landscape with mountains",
                            "a cute cat sitting on a table",
                            "a futuristic city at night",
                            "a portrait of a woman",
                            "a cup of coffee on a wooden table"
                        ],
                        help="Prompts for generation")
    parser.add_argument("--image_resolution", type=int, default=512,
                        help="Image resolution")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        help="HuggingFace model ID")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps for teacher model")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--scheduler", type=str, default="ddim",
                        choices=["ddim", "ddpm", "euler", "lcm"],
                        help="Scheduler type (use 'lcm' only for few-step sampling like 4-8 steps)")
    parser.add_argument("--lcm_timesteps", type=int, nargs="+", default=None,
                        help="Fixed timesteps for LCM (e.g., 999 749 499 249). Only used with --scheduler lcm")
    
    args = parser.parse_args()
    
    # global random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("Teacher Model Inference for SD1.5 - Standard Multi-Step Sampling")
    print("=" * 80)
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Number of images: {len(args.prompts)}")
    print(f"Random seed: {args.seed}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Scheduler: {args.scheduler}")
    if args.scheduler == "lcm" and args.lcm_timesteps:
        print(f"LCM timesteps: {args.lcm_timesteps}")
    print(f"Image resolution: {args.image_resolution}x{args.image_resolution}")
    print("=" * 80)
    
    # suggested configurations
    print("\n💡 Configuration Suggestions:")
    print("  Standard quality (50 steps):   --scheduler ddim --num_inference_steps 50 --guidance_scale 7.5")
    print("  Fast quality (25 steps):       --scheduler euler --num_inference_steps 25 --guidance_scale 7.5")
    print("  Few-step comparison (4 steps): --scheduler lcm --num_inference_steps 4 --lcm_timesteps 999 749 499 249 --guidance_scale 0")
    print("=" * 80)
    
    # load the teacher pipeline
    print("\nLoading teacher model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # sanity-check the scheduler / step-count combination
    if args.scheduler == "lcm" and args.num_inference_steps > 10:
        print("\n" + "="*80)
        print("⚠️  WARNING: LCM scheduler is designed for few-step sampling (4-8 steps)")
        print(f"   You are using {args.num_inference_steps} steps, which may produce suboptimal results.")
        print("   Recommendation: Use --scheduler ddim or --scheduler euler for >10 steps")
        print("="*80 + "\n")
    
    if args.scheduler != "lcm" and args.lcm_timesteps:
        print("\n" + "="*80)
        print("⚠️  WARNING: --lcm_timesteps is only used with --scheduler lcm")
        print("   Your timesteps setting will be ignored.")
        print("="*80 + "\n")
    
    # set the scheduler
    if args.scheduler == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        print("Using DDIM scheduler (good for 20-50 steps)")
    elif args.scheduler == "ddpm":
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        print("Using DDPM scheduler (good for 50-1000 steps)")
    elif args.scheduler == "euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        print("Using Euler scheduler (good for 20-50 steps)")
    elif args.scheduler == "lcm":
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        print("Using LCM scheduler (optimized for 4-8 steps)")
        if args.lcm_timesteps:
            print(f"  → Fixed timesteps: {args.lcm_timesteps}")
        else:
            print(f"  → Auto timesteps for {args.num_inference_steps} steps")
    
    pipeline = pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=False)
    
    # generate images
    print("\nGenerating images with teacher model...")
    images = generate_images_teacher(pipeline, args.prompts, args)
    
    # save images
    print("\nSaving images...")
    save_images(images, args.prompts, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
