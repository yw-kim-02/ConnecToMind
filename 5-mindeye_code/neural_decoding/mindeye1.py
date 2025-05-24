import os
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import PIL
from functools import partial
import glob

# for clip
import clip # OpenAI CLIP (RN50, ViT-L/14 등)
import open_clip # OpenAI CLIP (RN50, ViT-L/14 등) + LAION-5B로 학습한 모델
from transformers import CLIPVisionModelWithProjection

# for prior
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists
from tqdm.auto import tqdm
import random
import json
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig

# for vd 
from dalle2_pytorch.dalle2_pytorch import RotaryEmbedding, CausalTransformer, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward

# for low-level
from diffusers import DiffusionPipeline
from convnext import ConvnextXL
from diffusers.models.autoencoder_kl import Decoder

# get model
import utils
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
from schedulers import get_scheduler

class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False, hidden_state=False, device=torch.device('cpu')):
        super().__init__()
        self.device= device
        self.hidden_state = hidden_state
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)

        # clip preporcess를 custom으로 사용함
        self.preprocess = None # object를 변수로 저장
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073]) # OpenAI CLIP이 학습한 이미지 데이터 셋의 평균
        self.std = np.array([0.26862954, 0.26130258, 0.27577711]) # OpenAI CLIP이 학습한 이미지 데이터 셋의 표준편차
        self.normalize = transforms.Normalize(self.mean, self.std) # versatile low image vae 사용하기 전 normalize
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        preprocess = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.preprocess = preprocess # object를 변수로 저장

        # embedding preprocess 변수
        self.clamp_embs = clamp_embs # embdding 후처리 유무 ex) -1.5 ~ 1.5 범위로 제한
        self.norm_embs = norm_embs # embdding normalization 유무
        
        # "RN50", "ViT-L/14", "ViT-B/32", "RN50x64" 중에 모델이 없으면 오류메세지 출력
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64" # assert문은 조건을 만족하지 않을 때 출력
        print(clip_variant, device)

        # 1번 clip 모델 load(high level)
        if clip_variant=="ViT-L/14" and hidden_state:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval() # pre-train(transformers 라이브러리의 CLIP Vision 모델)하여 frozen 함
            image_encoder = image_encoder.to(device)
            # frozen
            for param in image_encoder.parameters():
                param.requires_grad = False 
            self.image_encoder = image_encoder
        elif hidden_state:
            raise Exception("ViT-L/14에서만 hidden_state 처리 가능")
        
        # 2번 clip 모델 load
        clip_model, _ = clip.load(clip_variant, device=device)
        clip_model.eval() 
        # frozen
        for param in clip_model.parameters():
            param.requires_grad = False 
        self.clip = clip_model

    def embed_image(self, image):
        '''
        clip_emb(hidden_state 없을 때): 최종 image embedding - shape: [batch, 768]
        clip_emb(hidden_state 있을 때): 최종 image embedding - shape: [batch, 257, 768]
            Tip: sequence_length는 vit의 sequence의 patch 개수
        '''
        if self.hidden_state:
            clip_emb = self.preprocess((image).to(self.device)) # image preprocess ex) image값 -1 ~ 1
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self._versatile_normalize_embeddings(clip_emb)
        else:
            clip_emb = self.preprocess(image.to(self.device)) # image preprocess
            clip_emb = self.clip.encode_image(clip_emb)

        # embedding processes: clamp_embs + norm_embs
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5) # embedding 값 -1.5 ~ 1.5 범위로 제한
        if self.norm_embs:
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
    
        return clip_emb

    def _versatile_normalize_embeddings(self, encoder_output):
        '''
        embeds(hidden_state 있을 때): image embedding normalization - shape: [batch, 257, 768]
            Tip: sequence_length는 vit의 sequence의 patch 개수
        '''
        embeds = encoder_output.last_hidden_state
        embeds = self.image_encoder.vision_model.post_layernorm(embeds)
        embeds = self.image_encoder.visual_projection(embeds)
        return embeds 
    
    
class BrainNetwork(nn.Module):
    def __init__(self, in_dim=15724, out_dim=257*768, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False):
        super().__init__()
        
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h) # batch norm과 layer norm에 인자(h)를 미리 고정
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU # batch norm이면 ReLU사용하도록 고정
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.clip_size = clip_size
        self.n_blocks = n_blocks

        # MLP back born 할 때 사용
        self.lin0 = nn.Sequential( # 15724 -> 4096
            nn.Linear(in_dim, h),
            *[item() for item in act_and_norm], # [nn.BatchNorm1d(4096), nn.ReLU(inplace=True)] -> unpacking
            nn.Dropout(0.5),
        ) 
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm], # [nn.BatchNorm1d(4096), nn.ReLU(inplace=True)] -> unpacking
                nn.Dropout(0.15)
            ) for _ in range(n_blocks) # 4개의 block 사용 -> sequential 4번 사용
        ])
        self.lin1 = nn.Linear(h, out_dim, bias=True) # 4096 -> (257 * 768)
        
        # contrastive learning 할 때 사용
        # clip_size -> 고차원 공간(2048) -> clip_size
        self.projector = nn.Sequential(
            nn.LayerNorm(clip_size),
            nn.GELU(),
            nn.Linear(clip_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, clip_size)
        )
        
    def forward(self, x):
        '''
        x(MLP backbone): fmri -> mlp - shape: [batch, (257*768)]
        x(MLP projector): fmri -> mlp -> mlp - shape: ([batch, 768], [batch, 257, 768])
        '''
        # fMRI volume 그대로 들어올 때
        if x.ndim == 4:
            # assert x.shape[1] == 81 and x.shape[2] == 104 and x.shape[3] == 83, "fMRI data shape 안 맞음" # [N, 81, 104, 83]은 nsd genaral roi이다
            assert x.shape[1] == 120 and x.shape[2] == 120 and x.shape[3] == 84, "fMRI data shape 안 맞음" # [N, 120, 120, 84]은 nsd raw data이다.
            x = x.reshape(x.shape[0], -1) # [N, 1209600]
        
        # MLP back born
        x = self.lin0(x) # bs, h

        residual = x
        for res_block in range(self.n_blocks): # block 개수 4개
            x = self.mlp[res_block](x)
            x += residual                                                                         
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)

        # MLP backborn + MLP projection(contrastive learning)
        return x, self.projector(x.reshape(len(x), -1, self.clip_size))
        
# BrainNetwork과 versatileDiffusionPriorNetwork가 인자로 들어감
class BrainDiffusionPrior(DiffusionPrior):
    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None) # 부모(DiffusionPrior)는 voxel2clip을 모르기 때문에 빼놓음
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    def forward(self, text_embed = None, image_embed = None, *args, **kwargs):
        '''
        loss: loss(prediction x_0, target x_0) - shape: Scala
        pred: x_t -> x_t-1 denoise한 결과 - shape: [batch, 768]
        '''
        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # ex) ex: 1.0 ~ 2.5 -> prediction 후 self.image_embed_scale을 나눠줘야 함
        image_embed *= self.image_embed_scale

        # calculate forward loss
        loss, pred = self.p_losses(image_embed=image_embed, times=times, text_embed=text_embed, *args, **kwargs)
        
        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred

    def p_losses(self, image_embed, times, text_embed=None, noise = None):
        '''
        loss: loss(prediction x_0, target x_0) - shape: Scala
        pred: x_t -> x_t-1 denoise한 결과 - shape: [batch, 768]
        '''
        # noise 정의
        noise = default(noise, lambda: torch.randn_like(image_embed)) 

        # random한 t가 들어옴 -> x_0에서 한 번에 t까지의 noise를 씌움 
        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        # self conditioning: prediction 값(x_0) = function(x_t, image_embed_noisy(첫 번째 prediction값)) 
        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                # self.net은 x_0의 prediction 값을 반환
                self_cond = self.net(image_embed_noisy, times, text_embed=text_embed).detach() # 첫 번째 prediction값을 뽑기위해 self.net사용

        # prediction 값(x_0)
        pred = self.net(image_embed_noisy, times, self_cond = self_cond, text_embed=text_embed, text_cond_drop_prob = self.text_cond_drop_prob, image_cond_drop_prob = self.image_cond_drop_prob) # cond_drop_prob: 텍스트 & 이미지 드롭아웃 확률 

        # prediction 값(x_0) normalization
        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        # x_0를 비교하는 방식
        target = image_embed

        loss = self.noise_scheduler.loss_fn(pred, target) # diffusion prior에서는 ε가 아닌 x_0를 예측함
        return loss, pred # train 할 때 보통 loss와 prediction 같이 반환

    @torch.no_grad()
    def p_sample_loop(self, shape, text_cond, cond_scale = 1., generator=None):
        batch, device = shape[0], self.device
        x_start = None 

        # image_embed = x_t(가우시안 노이즈 최종버젼)에서 시작
        if generator is None:
            image_embed = torch.randn(shape, device = device)
        else:
            image_embed = torch.randn(shape, device = device, generator=generator)

        # image_embedding(x_t) nomalization
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        # t_t -> x_0으로 denoise 시작
        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, self_cond = self_cond, cond_scale = cond_scale, generator=generator)
        
        # image embedding(x_0) nomalization
        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    # 한 스텝 denoise 적용
    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1., generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        
        # generator를 사용하면 분포에서 고정된 값을 뽑음 -> 매번 같은 이미지 생성
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)

        # t-1에 denoise 
        # 생성
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) # x_0이면 noise를 사용하지 않음
        # 적용
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start


# dalle2의 denoising에서 입실론을 뽑는 class(이름이 versatile인 이유는 모르겠음) -> DiffusionPrior class에서 net의 인자로 사용
class VersatileDiffusionPriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps = None, # time step 수 ex) 1000
        num_time_embeds = 1, # time embedding 개수
        num_tokens = 257,
        causal = True,
        learned_query_mode = 'none', # query 사용 방법
        **kwargs
    ):
        super().__init__()
        self.dim = dim # 768
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps) # time embedding값이 continous or descrete
        self.learned_query_mode = learned_query_mode # query가 학습되는 방법법

        # time embedding 변환
        self.to_time_embeds = nn.Sequential( 
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # query(pure noise로 만들어짐)
        if self.learned_query_mode == 'pos_emb': # image embedding + positional embedding
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)

        # transformer 저장
        self.causal_transformer = FlaggedCausalTransformer(dim = dim, causal=causal, **kwargs)

        # Classifier-Free Guidance: condition이 있을 때와 없을 때의 차이를 계산하기위해 사용되는 의미없는 embedding
        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        # tokens.shape = [batch_size, total_token, dim]
        # total_token= [ brain_embed(257개), time_embed(1개), image_embed(257개) , learned_queries(257개)] -> 이 중에서 맞춰야 할 embed
        self.num_tokens = num_tokens # 예측해야할 vector 차원
        
        # Self-conditioning: output을 input으로 사용하하여 output을 또 뽑음
        self.self_cond = False 

    # Classifier-Free Guidance(inference할 때만 사용)
    def forward_with_cond_scale(self,*args,cond_scale = 1.,**kwargs):
        # 모든 데이터가 condition이 적용된 prediction 값 - pure noise + cross attention
        logits = self.forward(*args, **kwargs) 

        if cond_scale == 1:
            return logits

        # 모든 데이터가 condition이 적용되지 않은 prediction 값 - pure noise
        null_logits = self.forward(*args, brain_cond_drop_prob = 1, image_cond_drop_prob = 1, **kwargs) # mask가 모두 1

        # condition이 있을 때와 없을 때의 차이를 계산 
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed, # query: pure noise 
        diffusion_timesteps,
        *,
        self_cond=None,
        brain_embed=None,
        text_embed=None,
        brain_cond_drop_prob = 0., # dropout 확률
        text_cond_drop_prob = None, # dropout 확률
        image_cond_drop_prob = 0. # dropout 확률
    ):  
        # brain embed = text embed  
        if text_embed is not None:
            brain_embed = text_embed 
        if text_cond_drop_prob is not None: 
            brain_cond_drop_prob = text_cond_drop_prob 
        
        # shape 변경
        image_embed = image_embed.view(len(image_embed),-1,768) # image_embed.shape -> [B, 257, 768]
        brain_embed = brain_embed.view(len(brain_embed),-1,768) # brain_embed.shape -> [B, 257, 768]
        
        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        
        # classifier free guidance masks 생성: inference에서는 모두 1인것과 모두 0인것을 둘 다 뽑음
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device = device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1') # brain_mask.shape -> [B, 1, 1]

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device = device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1') # image_mask.shape -> [B, 1, 1]


        # image_keep_mask에 해당하는 embedding값은 모두 0 or 1
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(
            brain_keep_mask, # broadcast
            brain_embed,
            null_brain_embeds[None]
        )

        # image_keep_mask에 해당하는 embedding값은 모두 0 or 1
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where( 
            image_keep_mask, # broadcast
            image_embed,
            null_image_embed[None]
        )

        # time_embedding
        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        # query 정의
        if self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch) # repeat: '257 768' shape -> 'b 257 768' shape로 변경
            image_embed = image_embed + pos_embs # query(pure noise로 만들어짐) -(forward)-> prediction 값

            tokens = torch.cat((
                brain_embed,  # key, value로 사용: [b, 257, 768] 
                time_embed,  # [b, 1, 768]
                image_embed,  # query로 사용: [b, 257, 768]
            ), dim = -2) # [b, 257 + 1 + 257 + 257, 768] = [b, 772, 768]

        # transformer
        tokens = self.causal_transformer(tokens) # output: [b, 772, 768]

        pred_image_embed = tokens[..., -self.num_tokens:, :] # output: [b, 257, 768] - image_embed를 prediction값으로 사용

        return pred_image_embed

class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth, # transformer의 layer 수 -> versatile의 한 step의 layer
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb), # tokens에서 key, query, value로 나눠짐
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x # attention + residual
            x = ff(x) + x # feedfoward + residual

        out = self.norm(x)
        return self.project_out(out)
    
class Voxel2StableDiffusionModel(torch.nn.Module):
    def __init__(self, in_dim=15724, h=4096, n_blocks=4, use_cont=True, ups_mode='4x'):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])

        # up ampling
        self.ups_mode = ups_mode 
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False) # 16384 = 64 * 16 * 16
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

    def forward(self, x, return_transformer_feats=False):
        '''
        loss 계산에 upsampler와 maps_projector 둘 다 필요하다

        self.upsampler(4x,8x): mlp -> up sampling - shape: [batch_size, 4, 64, 64] 
        self.maps_projector(x).flatten(2).permute(0,2,1): mlp -> up sampling -> projection - shape: 4x-[batch_size, 256, 512], 8x-[batch_size, 64, 256]
        '''

        # MLP = lin0 + lin1
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # (b,4096) -> (b,16384)

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)

class OpenClipper(torch.nn.Module):
    def __init__(self, clip_variant='ViT-H-14', hidden_state=False, norm_embs=False, device=torch.device('cpu')):
        super().__init__()
        print(clip_variant, device)
        assert clip_variant == 'ViT-H-14' # not setup for other models yet
         
        try:
            clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', 
                                        pretrained="/fsx/proj-medarc/fmri/cache/openclip/open_clip_pytorch_model.bin", device=device)
        except:
            print("no cached model found, downloading...")
            clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', 
                                        pretrained='laion2b_s32b_b79k', device=device)
            
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        # overwrite preprocess to accept torch inputs instead of PIL Image
        preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
            
        self.clip = clip_model
        self.preprocess = preprocess
        self.device = device
        self.norm_embs = norm_embs
        
        if hidden_state:
            print("THIS IS NOT WORKING CURRENTLY!")
            clip_model.visual.transformer.resblocks[31].mlp = nn.Identity()
            clip_model.visual.ln_post = nn.Identity()
            clip_model.token_embedding = nn.Identity()
            clip_model.ln_final = nn.Identity()
            
    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        clip_emb = self.preprocess(image.to(self.device))
        clip_emb = self.clip.encode_image(clip_emb)
        if self.norm_embs:
            clip_emb = nn.functional.normalize(clip_emb.flatten(1), dim=-1)
            clip_emb = clip_emb.reshape(len(clip_emb),-1,1024)
        return clip_emb


class BrainDiffusionPriorOld(DiffusionPrior):
    """ 
    Differences from original:
    - Allow for passing of generators to torch random functions
    - Option to include the voxel2clip model and pass voxels into forward method
    - Return predictions when computing loss
    - Load pretrained model from @nousr trained on LAION aesthetics
    """
    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1.,
                generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            #noise = torch.randn_like(x)
            noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, text_cond, cond_scale = 1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device = device)
        else:
            image_embed = torch.randn(shape, device = device, generator=generator)
        x_start = None # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, self_cond = self_cond, cond_scale = cond_scale, 
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond = self_cond,
            text_cond_drop_prob = self.text_cond_drop_prob,
            image_cond_drop_prob = self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss, pred

    def forward(
        self,
        text = None,
        image = None,
        voxel = None,
        text_embed = None,      # allow for training on preprocessed CLIP text and image embeddings
        image_embed = None,
        text_encodings = None,  # as well as CLIP text encodings
        *args,
        **kwargs
    ):
        # 
        assert exists(text) ^ exists(text_embed) ^ exists(voxel), 'either text, text embedding, or voxel must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(voxel):
            assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
            assert not exists(text_embed), 'cannot pass in both text and voxels'
            text_embed = self.voxel2clip(voxel)

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # scale image embed (Katherine)

        image_embed *= self.image_embed_scale

        # calculate forward loss

        loss, pred = self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)

        return loss, pred#, text_embed
   
    @staticmethod
    def from_pretrained(net_kwargs={}, prior_kwargs={}, voxel2clip_path=None, ckpt_dir='./checkpoints'):
        # "https://huggingface.co/nousr/conditioned-prior/raw/main/vit-l-14/aesthetic/prior_config.json"
        config_url = os.path.join(ckpt_dir, "prior_config.json")
        config = json.load(open(config_url))
        
        config['prior']['net']['max_text_len'] = 256
        config['prior']['net'].update(net_kwargs)
        # print('net_config', config['prior']['net'])
        net_config = DiffusionPriorNetworkConfig(**config['prior']['net'])

        kwargs = config['prior']
        kwargs.pop('clip')
        kwargs.pop('net')
        kwargs.update(prior_kwargs)
        # print('prior_config', kwargs)

        diffusion_prior_network = net_config.create()
        diffusion_prior = BrainDiffusionPriorOld(net=diffusion_prior_network, clip=None, **kwargs).to(torch.device('cpu'))
        
        # 'https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/aesthetic/best.pth'
        ckpt_url = os.path.join(ckpt_dir, 'best.pth')
        ckpt = torch.load(ckpt_url, map_location=torch.device('cpu'))

        # Note these keys will be missing (maybe due to an update to the code since training):
        # "net.null_text_encodings", "net.null_text_embeds", "net.null_image_embed"
        # I don't think these get used if `cond_drop_prob = 0` though (which is the default here)
        diffusion_prior.load_state_dict(ckpt, strict=False)
        # keys = diffusion_prior.load_state_dict(ckpt, strict=False)
        # print("missing keys in prior checkpoint (probably ok)", keys.missing_keys)

        if voxel2clip_path:
            # load the voxel2clip weights
            checkpoint = torch.load(voxel2clip_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['model_state_dict']
            for key in list(state_dict.keys()):
                if 'module.' in key:
                    state_dict[key.replace('module.', '')] = state_dict[key]
                    del state_dict[key]
            diffusion_prior.voxel2clip.load_state_dict(state_dict)
        
        return diffusion_prior

def get_model_lowlevel(args):

    if args.subj == 1:
        num_voxels = 15724
    elif args.subj == 2:
        num_voxels = 14278
    elif args.subj == 3:
        num_voxels = 15226
    elif args.subj == 4:
        num_voxels = 13153
    elif args.subj == 5:
        num_voxels = 13039
    elif args.subj == 6:
        num_voxels = 17907
    elif args.subj == 7:
        num_voxels = 12682
    elif args.subj == 8:
        num_voxels = 14386

    #### voxel2autoencoder ####
    # voxel2autoencoder 정의
    voxel2sd = Voxel2StableDiffusionModel(in_dim=num_voxels)

    # convnext + mlp 정의(loss에서만 사용)
    cnx_path = os.path.join(args.cache_dir, "convnext_xlarge_alpha0.75_fullckpt.pth")
    cnx = ConvnextXL(cnx_path)
    cnx.requires_grad_(False)
    cnx.eval()

    #### encoder & decoder 정의 ####
    try:
        # 전체 pipeline 로컬에서 로드
        sd_model_dir = os.path.join(args.cache_dir, "models--lambdalabs--sd-image-variations-diffusers", "snapshots")
        snapshot_name = os.listdir(sd_model_dir)  # 첫 snapshot
        snapshot_path = os.path.join(sd_model_dir, snapshot_name[0])

        sd_pipe = DiffusionPipeline.from_pretrained(snapshot_path)
    except Exception as e:
        print(f"[!] 로컬 snapshot 로딩 실패, 온라인에서 전체 모델 로드: {e}")
        sd_pipe = DiffusionPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", cache_dir=args.cache_dir)

    # 학습 비활성화
    sd_pipe.vae.eval()
    sd_pipe.vae.requires_grad_(False)
    
    vae = sd_pipe.vae
    noise_scheduler = sd_pipe.scheduler

    models = {
        "voxel2sd": voxel2sd,
        "cnx": cnx,
        "vae": vae,
        "noise_scheduler": noise_scheduler
    }

    return models

def get_model_highlevel(args):

    if args.subj == 1:
        num_voxels = 15724
    elif args.subj == 2:
        num_voxels = 14278
    elif args.subj == 3:
        num_voxels = 15226
    elif args.subj == 4:
        num_voxels = 13153
    elif args.subj == 5:
        num_voxels = 13039
    elif args.subj == 6:
        num_voxels = 17907
    elif args.subj == 7:
        num_voxels = 12682
    elif args.subj == 8:
        num_voxels = 14386
    
    #### clip 정의 ####
    clip_extractor = Clipper(clip_variant=args.clip_variant, norm_embs=args.norm_embs, hidden_state=args.hidden, device=args.device)
    
    #### brain network 정의 ####
    out_dim = args.token_size * args.clip_size # 257 * 768
    voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim, clip_size=args.clip_size)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)

    #### difussion prior 정의 ####
    out_dim = args.clip_size # 모델 거치면 257 * 768 나옴
    depth = 6 # transformer의 layer 수 -> versatile의 한 step의 layer
    dim_head = 64 # head당 dim
    heads = args.clip_size//dim_head # attention head 수
    timesteps = 100 # difusion step 수 
    
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = 257,
        learned_query_mode="pos_emb"
    ).to(args.device)

    # VersatileDiffusionPriorNetwork + BrainNetwork가 인자로 들어감
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network, # VersatileDiffusionPriorNetwork(필수 모델) -> nn.Module이라 학습 됨
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip, # BrainNetwork(추가 모델) -> nn.Module이라 학습 됨
    ).to(args.device)

    #### versatile difussion 정의(unet, vae, noise scheduler) ####
    try:
        # vd_model_dir = os.path.join(args.cache_dir, "models--shi-labs--versatile-diffusion")
        vd_model_dir = os.path.join(args.cache_dir, "models--shi-labs--versatile-diffusion", "snapshots")
        snapshot_name = os.listdir(vd_model_dir)  # 첫 snapshot
        snapshot_path = os.path.join(vd_model_dir, snapshot_name[0])
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_model_dir)
    except: # 처음에는 모델 불러와야 함
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion", cache_dir = args.cache_dir)
    
    # versatile difussion의 unet 정의
    vd_pipe.image_unet.eval()
    vd_pipe.image_unet.requires_grad_(False)
    # DualTransformer2DModel 이름 추출에서 image부분만 사용한다고 명시
    for name, module in vd_pipe.image_unet.named_modules(): # class vd_pipe.image_unet를 찍은 object들 
        if isinstance(module, DualTransformer2DModel): # object들 중 DualTransformer2DModel 이름 추출
            module.mix_ratio = 0.0 # versatile에서 image condition만 사용
            # text contex를 사용하지 않더라도 shape는 맞춰야 함
            for i, type in enumerate(("text", "image")):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    # versatile difussion의 vae 정의 -> inference의 decoder로 사용
    vd_pipe.vae.eval()
    vd_pipe.vae.requires_grad_(False)

    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler

    models = {
        "clip": clip_extractor,
        "diffusion_prior": diffusion_prior,
        "unet": unet, # inference에서만 사용
        "vae": vae, # inference에서만 사용
        "noise_scheduler": noise_scheduler, # inference에서만 사용
    }

    return models

