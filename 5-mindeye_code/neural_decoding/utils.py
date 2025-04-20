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
