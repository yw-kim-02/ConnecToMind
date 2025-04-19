import torch

def adamw(parameters, lr):
    return torch.optim.AdamW(parameters, lr=lr)

def get_optimizer(diffusion_prior, lr, optimizer_name='adamw'):

    # '일반 weigth'만 weight_decay를 주고 'bias', 'LayerNorm.bias', 'LayerNorm.weight'에는 주지 않는다.
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {
            'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 1e-2
        },
        {
            'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 1e-2
        },
        {
            'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    if optimizer_name.lower() == 'adamw':
        return adamw(opt_grouped_parameters, lr=lr)