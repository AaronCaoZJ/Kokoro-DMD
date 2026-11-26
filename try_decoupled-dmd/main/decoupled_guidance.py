from main.sd_guidance import SDGuidance, predict_noise
from main.utils import get_x0_from_noise
import torch
import torch.nn.functional as F


def sample_ca_timesteps(gen_timesteps, min_step):
    """CA re-noising level, strictly cleaner than the generator's input level.

    Decoupled DMD (arXiv 2511.22677, Alg. 1) samples tau_CA ~ U(t, 1) in its
    t=0 noise / t=1 data convention, i.e. DDPM index in [min_step, t).
    Falls back to min_step when t <= min_step.
    """
    t = gen_timesteps.long()
    span = (t - min_step).clamp_min(0)
    offset = torch.floor(torch.rand(t.shape, device=t.device) * span).long()
    return torch.where(span > 0, min_step + offset, torch.full_like(t, min_step))


class SDDecoupledGuidance(SDGuidance):
    def __init__(self, args, accelerator):
        # Inherits real_unet, fake_unet, scheduler, the timestep range, etc. from SDGuidance.
        super().__init__(args, accelerator)

        # The CA term needs both the cond and uncond real predictions; predict_noise
        # returns a single tensor when guidance_scale <= 1.
        assert self.real_guidance_scale > 1, "Decoupled DMD needs --real_guidance_scale > 1 (CA term is (w-1)(cond-uncond))"
        if self.use_f_divergence:
            print("[decoupled] warning: --use_f_divergence is ignored by the Decoupled DMD loss")

    def compute_decoupled_loss(
        self, 
        latents,  # generator output x0 (generated_image)
        text_embedding, 
        uncond_embedding, 
        unet_added_conditions, 
        uncond_unet_added_conditions,
        timesteps = None  # DDPM timesteps of the generator input
    ):
        # generator output x0; kept differentiable so the losses below reach the generator
        pred_generator_image = latents
        batch_size = latents.shape[0]
        
        with torch.no_grad():
            if timesteps is None:
                raise ValueError("timesteps must be provided for Decoupled DMD loss computation")
            
            # CA: per-sample timestep in [min_step, t_i), i.e. cleaner than the generator input (smaller DDPM index = less noise)
            ca_timesteps = sample_ca_timesteps(timesteps, self.min_step)

            # DM: timesteps drawn uniformly from [min_step, max_step+1), as in standard DMD
            dm_timesteps = torch.randint(
                self.min_step,
                min(self.max_step+1, self.num_train_timesteps),
                [batch_size],
                device=latents.device,
                dtype=torch.long
            )

            # CA and DM are re-noised with independent noise
            ca_noisy_latents = self.scheduler.add_noise(latents, torch.randn_like(latents), ca_timesteps)
            dm_noisy_latents = self.scheduler.add_noise(latents, torch.randn_like(latents), dm_timesteps)

            # DM fake cond
            pred_dm_fake_cond_noise = predict_noise(
                self.fake_unet, dm_noisy_latents, text_embedding, uncond_embedding,
                dm_timesteps, guidance_scale=self.fake_guidance_scale,
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions,
                decoupled=False
            )
            pred_dm_fake_cond_image = get_x0_from_noise(
                dm_noisy_latents.double(), pred_dm_fake_cond_noise.double(), self.alphas_cumprod.double(), dm_timesteps
            )

            if self.use_fp16:
                if self.sdxl:
                    bf16_unet_added_conditions = {} 
                    bf16_uncond_unet_added_conditions = {} 

                    for k,v in unet_added_conditions.items():
                        bf16_unet_added_conditions[k] = v.to(torch.bfloat16)
                    for k,v in uncond_unet_added_conditions.items():
                        bf16_uncond_unet_added_conditions[k] = v.to(torch.bfloat16)
                else:
                    bf16_unet_added_conditions = unet_added_conditions 
                    bf16_uncond_unet_added_conditions = uncond_unet_added_conditions

                pred_ca_real_cond_noise, pred_ca_real_uncond_noise = predict_noise(
                    self.real_unet, ca_noisy_latents.to(torch.bfloat16), text_embedding.to(torch.bfloat16), 
                    uncond_embedding.to(torch.bfloat16), 
                    ca_timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=bf16_unet_added_conditions,
                    uncond_unet_added_conditions=bf16_uncond_unet_added_conditions,
                    decoupled=True
                )
                # DM only needs the real cond prediction: guidance_scale=1 runs the cond branch alone, skipping the uncond forward pass
                pred_dm_real_cond_noise = predict_noise(
                    self.real_unet, dm_noisy_latents.to(torch.bfloat16), text_embedding.to(torch.bfloat16),
                    uncond_embedding.to(torch.bfloat16),
                    dm_timesteps, guidance_scale=1.0,
                    unet_added_conditions=bf16_unet_added_conditions,
                    uncond_unet_added_conditions=bf16_uncond_unet_added_conditions
                )
            else:
                pred_ca_real_cond_noise, pred_ca_real_uncond_noise = predict_noise(
                    self.real_unet, ca_noisy_latents, text_embedding, uncond_embedding, 
                    ca_timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=unet_added_conditions,
                    uncond_unet_added_conditions=uncond_unet_added_conditions,
                    decoupled=True
                )
                pred_dm_real_cond_noise = predict_noise(
                    self.real_unet, dm_noisy_latents, text_embedding, uncond_embedding,
                    dm_timesteps, guidance_scale=1.0,
                    unet_added_conditions=unet_added_conditions,
                    uncond_unet_added_conditions=uncond_unet_added_conditions
                )

            # DM real cond
            pred_dm_real_cond_image = get_x0_from_noise(
                dm_noisy_latents.double(), pred_dm_real_cond_noise.double(), self.alphas_cumprod.double(), dm_timesteps
            )
            # CA real cond and uncond
            pred_ca_real_cond_image = get_x0_from_noise(
                ca_noisy_latents.double(), pred_ca_real_cond_noise.double(), self.alphas_cumprod.double(), ca_timesteps
            )
            pred_ca_real_uncond_image = get_x0_from_noise(
                ca_noisy_latents.double(), pred_ca_real_uncond_noise.double(), self.alphas_cumprod.double(), ca_timesteps
            )

            # CA update vector
            ca_update_vector = (self.real_guidance_scale - 1) * (pred_ca_real_cond_image - pred_ca_real_uncond_image)
            ca_norm_factor = (pred_ca_real_cond_image - pred_generator_image).abs().mean(dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
            ca_update_vector = ca_update_vector / (ca_norm_factor + 1e-8)
            
            # DM update vector
            dm_update_vector = (pred_dm_real_cond_image - pred_dm_fake_cond_image)
            dm_norm_factor = (pred_dm_real_cond_image - pred_generator_image).abs().mean(dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
            dm_update_vector = dm_update_vector / (dm_norm_factor + 1e-8)

        # Losses on pred_generator_image (= latents); gradients flow back to the generator
        loss_ca = F.mse_loss(pred_generator_image.float(), (pred_generator_image + ca_update_vector).detach().float(), reduction="mean")
        loss_dm = F.mse_loss(pred_generator_image.float(), (pred_generator_image + dm_update_vector).detach().float(), reduction="mean")

        loss = loss_dm + loss_ca

        loss_dict = {
            "loss_decoupled": loss,
            "loss_ca": loss_ca,
            "loss_dm": loss_dm
        }

        decoupled_log_dict = {
            "ca_noisy_latents": ca_noisy_latents.detach().float(),
            "dm_noisy_latents": dm_noisy_latents.detach().float(),
            "pred_ca_real_cond_image": pred_ca_real_cond_image.detach().float(),
            "pred_ca_real_uncond_image": pred_ca_real_uncond_image.detach().float(),
            "pred_dm_real_cond_image": pred_dm_real_cond_image.detach().float(),
            "pred_dm_fake_cond_image": pred_dm_fake_cond_image.detach().float(),
            "ca_update_vector": ca_update_vector.detach().float(),
            "dm_update_vector": dm_update_vector.detach().float(),
            "ca_norm_factor": ca_norm_factor.detach().float().mean(),  # batch mean
            "dm_norm_factor": dm_norm_factor.detach().float().mean(),  # batch mean
            "ca_update_norm": torch.norm(ca_update_vector).item(),  # norm of the update vector
            "dm_update_norm": torch.norm(dm_update_vector).item(),  # norm of the update vector
            "ca_timesteps_mean": ca_timesteps.float().mean().item(),  # mean CA timestep
            "dm_timesteps_mean": dm_timesteps.float().mean().item(),  # mean DM timestep
        }

        return loss_dict, decoupled_log_dict

    def generator_forward(
        self,
        image,
        text_embedding,
        uncond_embedding,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None,
        timesteps = None  # generator input timesteps (from backward simulation when enabled)
    ):
        loss_dict = {}
        log_dict = {}

        # Distribution matching loss, decoupled into the CA and DM terms
        if not self.gan_alone:
            dm_dict, dm_log_dict = self.compute_decoupled_loss(
                image, text_embedding, uncond_embedding, 
                unet_added_conditions, uncond_unet_added_conditions,
                timesteps=timesteps
            )

            loss_dict.update(dm_dict)
            log_dict.update(dm_log_dict)

        # Optional GAN generator loss from the classifier on the clean generated image
        if self.cls_on_clean_image:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(
                image, text_embedding, unet_added_conditions
            )
            loss_dict.update(clean_cls_loss_dict)

        return loss_dict, log_dict

    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None, 
        guidance_data_dict=None
    ):    
        if generator_turn:
            # Generator input timesteps; they bound the CA re-noising level
            timesteps = generator_data_dict.get("timesteps", None)
            
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict["image"],
                text_embedding=generator_data_dict["text_embedding"],
                uncond_embedding=generator_data_dict["uncond_embedding"],
                unet_added_conditions=generator_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=generator_data_dict["uncond_unet_added_conditions"],
                timesteps=timesteps
            )   
        elif guidance_turn:
            # The guidance turn (fake score + classifier) is unchanged from SDGuidance
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict["image"],
                text_embedding=guidance_data_dict["text_embedding"],
                uncond_embedding=guidance_data_dict["uncond_embedding"],
                real_train_dict=guidance_data_dict["real_train_dict"],
                unet_added_conditions=guidance_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=guidance_data_dict["uncond_unet_added_conditions"]
            ) 
        else:
            raise NotImplementedError

        return loss_dict, log_dict