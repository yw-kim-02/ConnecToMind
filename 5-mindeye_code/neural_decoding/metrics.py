import torch
import numpy as np
from torchvision import transforms
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity 

import clip
from torchvision.transforms.functional import to_pil_image
from torchvision.models import alexnet, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import AlexNet_Weights, Inception_V3_Weights


# PixCorr
def pixcorr(recons, gts, resize=425):
    preprocess = transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR)
    recons = preprocess(recons).view(len(recons), -1).float().cpu()
    gts = preprocess(gts).view(len(gts), -1).float().cpu()
    corrsum = sum(np.corrcoef(r, g)[0, 1] for r, g in zip(recons, gts))
    return corrsum / len(recons)

# SSIM
def ssim(recons, gts, resize=425):
    preprocess = transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR)
    recon_gray = rgb2gray(preprocess(recons).permute(0,2,3,1).cpu())
    gts_gray = rgb2gray(preprocess(gts).permute(0,2,3,1).cpu())
    ssim_scores = [
        structural_similarity(r, g, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        for r, g in zip(recon_gray, gts_gray)
    ]
    return np.mean(ssim_scores)

# Clip
def clip_metric(args, recons, gts, clip_model, preprocess):
    return two_way_identification(args, recons, gts, clip_model.encode_image, preprocess)

# AlexNet 2
def alexnet_2(args, recons, gts, alex_model, preprocess, layer='features.4'):
    return two_way_identification(args, recons, gts, alex_model, preprocess, layer)

# AlexNet 5
def alexnet_5(args, recons, gts, alex_model, preprocess, layer='features.11'):
    return two_way_identification(args, recons, gts, alex_model, preprocess, layer)

# Inception
def inception(args, recons, gts, incep_model, preprocess):
    return two_way_identification(args, recons, gts, incep_model, preprocess, 'avgpool')

# 2-way Identification (used for CLIP, AlexNet, Inception)
@torch.no_grad()
def two_way_identification(args, recons, gts, model, preprocess, feature_layer=None):
    device = args.device

    # (PIL or Tensor) + preprocess
    recon_images = [process_image(r, preprocess) for r in recons]
    gt_images    = [process_image(g, preprocess) for g in gts]

    pred_feats = model(torch.stack(recon_images).to(device))
    gt_feats = model(torch.stack(gt_images).to(device))

    if feature_layer:
        pred_feats = pred_feats[feature_layer].flatten(1).detach().cpu().numpy()
        gt_feats = gt_feats[feature_layer].flatten(1).detach().cpu().numpy()
    else:
        pred_feats = pred_feats.float().flatten(1).detach().cpu().numpy()
        gt_feats = gt_feats.float().flatten(1).detach().cpu().numpy()

    r = np.corrcoef(gt_feats, pred_feats)
    r = r[:len(gt_feats), len(gt_feats):]
    correct = (r < np.expand_dims(np.diag(r), axis=1)).sum(axis=1)
    return np.mean(correct) / (len(gt_feats) - 1)

def process_image(tensor_img, preprocess):
    """
    PIL → preprocess → Tensor (for CLIP)
    또는
    PIL → ToTensor → preprocess (for AlexNet/Inception)
    """
    pil_img = to_pil_image(tensor_img.cpu())
    try:
        return preprocess(pil_img)
    except Exception:
        tensor_img = transforms.ToTensor()(pil_img)
        return preprocess(tensor_img)

def get_metric(args):
    device = args.device
    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval().requires_grad_(False)

    # AlexNet
    alex = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
    alex_extractor = create_feature_extractor(alex, return_nodes={
        "features.4": "features.4",
        "features.11": "features.11",
    })
    alex_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Inception
    incep = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
    incep_extractor = create_feature_extractor(incep, return_nodes={"avgpool": "avgpool"})
    incep_preprocess = transforms.Compose([
        transforms.Resize(342),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 마지막에 딕셔너리로 모두 묶기
    metrics = {
        "pixcorr": pixcorr,
        "ssim": ssim,
        "clip": {
            "model": clip_model,
            "preprocess": clip_preprocess,
            "metric_fn": clip_metric,
        },
        "alexnet2": {
            "model": alex_extractor,
            "preprocess": alex_preprocess,
            "layer": "features.4",
            "metric_fn": alexnet_2,
        },
        "alexnet5": {
            "model": alex_extractor,
            "preprocess": alex_preprocess,
            "layer": "features.11",
            "metric_fn": alexnet_5,
        },
        "inception": {
            "model": incep_extractor,
            "preprocess": incep_preprocess,
            "metric_fn": inception,
        },
    }

    return metrics
    