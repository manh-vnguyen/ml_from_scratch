import torch
from torch import nn

class BatchNorm(nn.Module):
    VARIANCE_UNBIASED = True # False will make it identical to nn.LayerNorm
    def __init__(self, num_channels, momentum=0.1, eps=1e-5):
        super().__init__()
        self.repr = f"BatchNorm({num_channels=}, {momentum=}, {eps=})"
        self.momentum = momentum
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels).float())
        self.beta = nn.Parameter(torch.zeros(num_channels).float())
        self.register_buffer('running_mean', torch.zeros(num_channels)) 
        self.register_buffer('running_var', torch.ones(num_channels))

    def __repr__(self):
        return self.repr
    
    def forward(self, x): # (B, C, *)
        if self.training:
            mean = x.transpose(0, 1).mean(dim=list(range(1, x.ndim)), keepdim=True).transpose(0, 1) # (1, C, 1, 1)
            var = x.transpose(0, 1).var(dim=list(range(1, x.ndim)), keepdim=True, unbiased=self.VARIANCE_UNBIASED).transpose(0, 1) # (1, C, 1, 1)
            x = (x - mean) / torch.sqrt(var + self.eps) # transform the mean and var into (1, C, 1, 1) for broadcasting semanticj
            x = x * self.gamma.view([-1]+[1] * (x.ndim - 2)) + self.beta.view([-1]+[1] * (x.ndim - 2))
            self.running_mean = self.momentum * mean.view(-1) + (1-self.momentum) * self.running_mean
            self.running_var = self.momentum * var.view(-1) + (1-self.momentum) * self.running_var
        else:
            x = (x - self.running_mean.view([-1]+[1] * (x.ndim - 2))) / torch.sqrt(self.running_var.view([-1]+[1] * (x.ndim - 2)) + self.eps) # transform the mean and var into (1, C, 1, 1) for broadcasting semantic
            x = x * self.gamma.view([-1]+[1] * (x.ndim - 2)) + self.beta.view([-1]+[1] * (x.ndim - 2))
        return x


class BNTT(nn.Module):
    def __init__(self, T, num_channels):
        super().__init__()
        self.bntt = nn.ModuleList([BatchNorm(num_channels) for _ in range(T)])
        self.t = 0

    def reset(self):
        self.t = 0

    def forward(self, x):
        out = self.bntt[self.t](x)
        self.t += 1
        return out
    
def get_best_gpu(min_required):
    import os
    max_arg = lambda a: (max(a), a.index(max(a)))
    free, gpu = max_arg(list(map(int, os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits').read().strip().split('\n'))))
    if free >= min_required:
        return gpu
    raise Exception("Not enough resources")