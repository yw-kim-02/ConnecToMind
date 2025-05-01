import random
import os
import numpy as np
import torch

# image augmentation
import kornia
from kornia.augmentation.container import AugmentationSequential

# mixco
import torch.nn.functional as F

# seed
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

# image augmentation
def img_augment(image: torch.Tensor):
    img_augment_pipeline = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224,224), (0.6,1), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.3),
        data_keys=["input"],
    )

    augmented = img_augment_pipeline(image)

    return augmented

# mix up(배치에서 무작위로 고른 1개 샘플과만 섞음)
def mixup(fmri_vol, beta=0.15, s_thresh=0.5): # ex) fmri_vol.shape: [B, num_voxels]
    perm = torch.randperm(fmri_vol.shape[0])
    fmri_vol_shuffle = fmri_vol[perm].to(fmri_vol.device,dtype=fmri_vol.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([fmri_vol.shape[0]]).to(fmri_vol.device,dtype=fmri_vol.dtype)
    select = (torch.rand(fmri_vol.shape[0]) <= s_thresh).to(fmri_vol.device)
    betas_shape = [-1] + [1]*(len(fmri_vol.shape)-1)
    fmri_vol[select] = fmri_vol[select] * betas[select].reshape(*betas_shape) + \
        fmri_vol_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return fmri_vol, perm, betas, select # perm, betas, select는 loss계산할 때 사용

# mixco_nce loss
def mixco_nce_loss(prediction, target, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (prediction @ target.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(prediction.shape[0]).to(prediction.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss

# cosine annealing scheduling
def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

# soft_clip loss
def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

# top-k 중에 정답이 포함되어 있는지 판단
def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

# 상관계수 행렬처럼 target과 prediction간의 cosine_similarity 행렬 -> [target 개수, prediction 개수] 크기의 similarity matrix
def batchwise_cosine_similarity(Z,B):
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

# 파일 저장할때 뒤에 자동증가 숫자 붙이기
def get_unique_path(base_path):
    """
    중복되지 않는 파일 경로를 반환.
    예: diffusion_prior.pth → diffusion_prior_1.pth → diffusion_prior_2.pth ...
    """
    if not os.path.exists(base_path):
        return base_path

    base, ext = os.path.splitext(base_path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

# reconstruction
@torch.no_grad()
def reconstruction(
    brain_clip_embeddings, image,
    clip_extractor, unet=None, vae=None, noise_scheduler=None,
    seed = 42,
    num_inference_steps = 50,
    recons_per_sample = 1, # mindeye에서는 16개
    inference_batch_size=1, # batch 중에서 몇 개만 저장할지 -> batch와 같이 줄 것
    img_lowlevel = None, # low level image
    guidance_scale = 7.5,
    img2img_strength = .85,
    plotting=True,
):
    #### setting ####
    if unet:
        # CFG 사용
        do_classifier_free_guidance = guidance_scale > 1.0 
        # resolution 비율: down sampling, up sampling 비율율
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) 
        # 생성할 iamge resolution
        height = unet.config.sample_size * vae_scale_factor 
        width = unet.config.sample_size * vae_scale_factor

    #### versatile diffusion ####
    # cls token norm을 사용하여 모든 patch normalization
    for samp in range(len(brain_clip_embeddings)):
        brain_clip_embeddings[samp] = brain_clip_embeddings[samp]/(brain_clip_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)  
    
    # versatile difussion에 사용할 embedding 정의
    input_embedding = brain_clip_embeddings
    prompt_embeds = torch.zeros(len(input_embedding),77,768) # 사용하지 않을 거지만 difussion.unet.DualTransformer2DModel을 사용하기 위해 필요 -> 모두 0으로 채운 것을 넣기
    
    # CFG 준비
    if do_classifier_free_guidance:
        input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
        prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype) # [2 * b, 77, 768]
    
    # dual_prompt_embeddings
    input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1) # [2 * b, 257+77, 768]
    
    # CFG로 인해 batch size가 2배 늘어난 것을 2배 나눠야 한다
    shape = (inference_batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor) # [b, 257+77, 768]
    
    # timesteps 정의
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    
    # using low level image vs using pure noise
    if img_lowlevel is not None: # low level image에서 시작
        init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = noise_scheduler.timesteps[t_start:]
        latent_timestep = timesteps[:1].repeat(inference_batch_size)
        
        if verbose: print("img_lowlevel", img_lowlevel.shape)
        img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
        if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
        init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
        init_latents = vae.config.scaling_factor * init_latents
        init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

        noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                            generator=generator, dtype=input_embedding.dtype)
        init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
        latents = init_latents
    else: # pure noise에서 시작
        timesteps = noise_scheduler.timesteps
        latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                generator=generator, dtype=input_embedding.dtype)
        latents = latents * noise_scheduler.init_noise_sigma

    # inference - (Denoising loop) 
    for i, t in enumerate(timesteps):
        # cfg를 사용하기 위해 gaussian noise를 condition용 + uncondition용을 만든다
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
        # noise 예측
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_context = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_context - noise_pred_uncond)

        # compute denoise(x_t) -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    # vae decoder를 통해 image로 변환
    recons = decode_latents(latents,vae).detach().cpu()
    brain_recons = recons.view(inference_batch_size, recons_per_sample, 3, height, width) # [b*(recons_per_sample), 3, height, width] -> [b, recons_per_sample, 3, height, width]
    print("brain_recons",brain_recons.shape)
        
    #### pick best reconstruction out of several ####
    best_picks = np.zeros(inference_batch_size).astype(np.int16)  # best reconstruction 인덱스를 담은 vector
    v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings), -1), dim=-1)  # [inference_batch_size, clip_dim]
    for sample_idx in range(inference_batch_size):  # inference_batch_size
        sims = []
        reference = v2c_reference_out[sample_idx:sample_idx+1]  # [1, clip_dim]

        for recon_idx in range(recons_per_sample):
            currecon = brain_recons[sample_idx, recon_idx].unsqueeze(0).float()  # [1, 3, H, W]
            currecon = clip_extractor.embed_image(currecon).to(proj_embeddings.device).to(proj_embeddings.dtype)  # [1, clip_dim]
            currecon = nn.functional.normalize(currecon.view(len(currecon), -1), dim=-1)  # normalize

            cursim = batchwise_cosine_similarity(reference, currecon)  # (1, 1) similarity
            sims.append(cursim.item())  # scalar 값만 append

        best_picks[sample_idx] = int(np.nanargmax(sims))  # sample_idx 위치에 best recon index 저장
        print(f"Sample {sample_idx} sims: {sims}")
        print(f"Sample {sample_idx} best pick: {best_picks[sample_idx]}")

    #### plot ####
    img2img_samples = 0 if img_lowlevel is None else 1
    num_xaxis_subplots = 1 + img2img_samples + recons_per_sample
    best_img = torch.zeros((inference_batch_size, 3, height, width), dtype=brain_recons.dtype) # 초기화

    if plotting:
        fig, ax = plt.subplots(inference_batch_size, num_xaxis_subplots, 
                            figsize=(num_xaxis_subplots*5, 6*inference_batch_size),
                            facecolor=(1, 1, 1))
        for recon_idx in range(inference_batch_size):
            # ax가 1D array일 수도, 2D array일 수도 있음
            axis_row = ax[recon_idx] if inference_batch_size > 1 else ax

            axis_row[0].set_title(f"Original Image")
            axis_row[0].imshow(torch_to_Image(image[recon_idx]))

            if img2img_samples == 1:
                axis_row[1].set_title(f"Img2img ({img2img_strength})")
                axis_row[1].imshow(torch_to_Image(img_lowlevel[recon_idx].clamp(0, 1)))

            for ii, subplot_idx in enumerate(range(num_xaxis_subplots - recons_per_sample, num_xaxis_subplots)):
                recon = brain_recons[recon_idx][ii]
                if ii == best_picks[recon_idx]:
                    axis_row[subplot_idx].set_title(f"Reconstruction", fontweight='bold')
                    best_img[recon_idx] = recon
                else:
                    axis_row[subplot_idx].set_title(f"Recon {ii+1} from brain")
                axis_row[subplot_idx].imshow(torch_to_Image(recon))

            for subplot in axis_row:
                subplot.axis('off')
    else:
        fig = None
        best_img = brain_recons[range(inference_batch_size), best_picks]


    return fig,          # 전체 subplot figure
            brain_recons, # 모든 복원 결과 tensor ex) shape [b, recons_per_sample, 3, height, width]
            best_picks,   # best reconstruction 인덱스 ex) [0번 batch의 가장 좋은 index, 1번 batch의 가장 좋은 index, ...]
            best_img     # best reconstruction 이미지 ex) [0번 batch의 가장 좋은 image, 1번 batch의 가장 좋은 image, ...]



# # reconstruction
# @torch.no_grad()
# def reconstruction(
#     image, voxel,
#     clip_extractor, unet=None, vae=None, noise_scheduler=None,
#     diffusion_prior=None,
#     img_lowlevel = None, # low level image
#     num_inference_steps = 50,
#     recons_per_sample = 1, # mindeye에서는 16개
#     inference_batch_size=1, # batch 중에서 몇 개만 저장할지 -> batch와 같이 줄 것
#     guidance_scale = 7.5,
#     img2img_strength = .85,
#     timesteps_prior = 100,
#     seed = 0,
#     plotting=True,
# ):
#     #### setting ####
#     # batch 중에서 몇 개만 복원해서 저장할지 설정 ex) 1이면 첫 번째 사진만 저장
#     voxel=voxel[:inference_batch_size]
#     image=image[:inference_batch_size]

#     if unet:
#         # CFG 사용
#         do_classifier_free_guidance = guidance_scale > 1.0 
#         # resolution 비율: down sampling, up sampling 비율율
#         vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) 
#         # 생성할 iamge resolution
#         height = unet.config.sample_size * vae_scale_factor 
#         width = unet.config.sample_size * vae_scale_factor
    
#     # noise 난수 고정
#     generator = torch.Generator(device=device)
#     generator.manual_seed(seed)

#     #### MLP inference ####
#     brain_clip_embeddings0, proj_embeddings = diffusion_prior.voxel2clip(voxel)
#     brain_clip_embeddings0 = brain_clip_embeddings0.view(len(voxel),-1,768) # [B, (257 * 768)] -> [B, 257, 768]
#     proj_embeddings = proj_embeddings.cpu() # pick 할 때만 사용되어서 cpu로 내림
    
#     #### diffusion prior inference ####
#     brain_clip_embeddings0 = brain_clip_embeddings0.repeat_interleave(recons_per_sample, dim=0) # [B×(recons_per_sample), 257, 768] -> ex) 1번,1번,1번,2번,2번,2번,...
#     # diffusion prior 결과
#     brain_clip_embeddings = diffusion_prior.p_sample_loop(brain_clip_embeddings0.shape, 
#                             text_cond = dict(text_embed = brain_clip_embeddings0), 
#                             cond_scale = 1., timesteps = timesteps_prior,generator=generator) 

#     #### versatile diffusion ####
#     # cls token norm을 사용하여 모든 patch normalization
#     for samp in range(len(brain_clip_embeddings)):
#         brain_clip_embeddings[samp] = brain_clip_embeddings[samp]/(brain_clip_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)  
    
#     # versatile difussion에 사용할 embedding 정의
#     input_embedding = brain_clip_embeddings
#     prompt_embeds = torch.zeros(len(input_embedding),77,768) # 사용하지 않을 거지만 difussion.unet.DualTransformer2DModel을 사용하기 위해 필요 -> 모두 0으로 채운 것을 넣기
    
#     # CFG 준비
#     if do_classifier_free_guidance:
#         input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
#         prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype) # [2 * b, 77, 768]
    
#     # dual_prompt_embeddings
#     input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1) # [2 * b, 257+77, 768]
    
#     # CFG로 인해 batch size가 2배 늘어난 것을 2배 나눠야 한다
#     shape = (inference_batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor) # [b, 257+77, 768]
    
#     # timesteps 정의
#     noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    
#     # using low level image vs using pure noise
#     if img_lowlevel is not None: # low level image에서 시작
#         init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
#         t_start = max(num_inference_steps - init_timestep, 0)
#         timesteps = noise_scheduler.timesteps[t_start:]
#         latent_timestep = timesteps[:1].repeat(inference_batch_size)
        
#         if verbose: print("img_lowlevel", img_lowlevel.shape)
#         img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
#         if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
#         init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
#         init_latents = vae.config.scaling_factor * init_latents
#         init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

#         noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
#                             generator=generator, dtype=input_embedding.dtype)
#         init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
#         latents = init_latents
#     else: # pure noise에서 시작
#         timesteps = noise_scheduler.timesteps
#         latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
#                                 generator=generator, dtype=input_embedding.dtype)
#         latents = latents * noise_scheduler.init_noise_sigma

#     # inference - (Denoising loop) 
#     for i, t in enumerate(timesteps):
#         # cfg를 사용하기 위해 gaussian noise를 condition용 + uncondition용을 만든다
#         latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
#         latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
#         # noise 예측
#         noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

#         # perform guidance
#         if do_classifier_free_guidance:
#             noise_pred_uncond, noise_pred_context = noise_pred.chunk(2)
#             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_context - noise_pred_uncond)

#         # compute denoise(x_t) -> x_t-1
#         latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
#     # vae decoder를 통해 image로 변환
#     recons = decode_latents(latents,vae).detach().cpu()
#     brain_recons = recons.view(inference_batch_size, recons_per_sample, 3, height, width) # [b*(recons_per_sample), 3, height, width] -> [b, recons_per_sample, 3, height, width]
#     print("brain_recons",brain_recons.shape)
        
#     #### pick best reconstruction out of several ####
#     best_picks = np.zeros(inference_batch_size).astype(np.int16)  # best reconstruction 인덱스를 담은 vector
#     v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings), -1), dim=-1)  # [inference_batch_size, clip_dim]
#     for sample_idx in range(inference_batch_size):  # inference_batch_size
#         sims = []
#         reference = v2c_reference_out[sample_idx:sample_idx+1]  # [1, clip_dim]

#         for recon_idx in range(recons_per_sample):
#             currecon = brain_recons[sample_idx, recon_idx].unsqueeze(0).float()  # [1, 3, H, W]
#             currecon = clip_extractor.embed_image(currecon).to(proj_embeddings.device).to(proj_embeddings.dtype)  # [1, clip_dim]
#             currecon = nn.functional.normalize(currecon.view(len(currecon), -1), dim=-1)  # normalize

#             cursim = batchwise_cosine_similarity(reference, currecon)  # (1, 1) similarity
#             sims.append(cursim.item())  # scalar 값만 append

#         best_picks[sample_idx] = int(np.nanargmax(sims))  # sample_idx 위치에 best recon index 저장
#         print(f"Sample {sample_idx} sims: {sims}")
#         print(f"Sample {sample_idx} best pick: {best_picks[sample_idx]}")

#     #### plot ####
#     img2img_samples = 0 if img_lowlevel is None else 1
#     num_xaxis_subplots = 1 + img2img_samples + recons_per_sample

#     if plotting:
#         fig, ax = plt.subplots(inference_batch_size, num_xaxis_subplots, 
#                             figsize=(num_xaxis_subplots*5, 6*inference_batch_size),
#                             facecolor=(1, 1, 1))
#         for recon_idx in range(inference_batch_size):
#             # ax가 1D array일 수도, 2D array일 수도 있음
#             axis_row = ax[recon_idx] if inference_batch_size > 1 else ax

#             axis_row[0].set_title(f"Original Image")
#             axis_row[0].imshow(torch_to_Image(image[recon_idx]))

#             if img2img_samples == 1:
#                 axis_row[1].set_title(f"Img2img ({img2img_strength})")
#                 axis_row[1].imshow(torch_to_Image(img_lowlevel[recon_idx].clamp(0, 1)))

#             for ii, subplot_idx in enumerate(range(num_xaxis_subplots - recons_per_sample, num_xaxis_subplots)):
#                 recon = brain_recons[recon_idx][ii]
#                 if ii == best_picks[recon_idx]:
#                     axis_row[subplot_idx].set_title(f"Reconstruction", fontweight='bold')
#                     if recon_idx == 0:  # 첫 번째 샘플의 best pick 저장
#                         best_img = recon
#                 else:
#                     axis_row[subplot_idx].set_title(f"Recon {ii+1} from brain")
#                 axis_row[subplot_idx].imshow(torch_to_Image(recon))

#             for subplot in axis_row:
#                 subplot.axis('off')
#     else:
#         fig = None
#         best_img = brain_recons[range(inference_batch_size), best_picks]


#     return fig,          # 전체 subplot figure
#             brain_recons, # 모든 복원 결과 tensor ex) shape [b, recons_per_sample, 3, height, width]
#             best_picks,   # best reconstruction 인덱스 ex) [0번 batch의 가장 좋은 index, 1번 batch의 가장 좋은 index, ...]
#             best_img     # best reconstruction 이미지 ex) [0번 batch의 가장 좋은 image, 1번 batch의 가장 좋은 image, ...]