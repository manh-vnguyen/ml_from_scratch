#%%
import torch
from torch import nn
from einops import rearrange, reduce


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
    VARIANCE_UNBIASED = True # False will make it identical to nn.LayerNorm
    def __init__(self, T, num_channels, momentum=0.1, eps=1e-5):
        super().__init__()
        self.repr = f"BatchNorm({T=}, {num_channels=}, {momentum=}, {eps=})"
        self.momentum = momentum
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(T, num_channels).float())
        self.beta = nn.Parameter(torch.zeros(T, num_channels).float())
        self.register_buffer('running_mean', torch.zeros(T, num_channels)) 
        self.register_buffer('running_var', torch.ones(T, num_channels))

    def __repr__(self):
        return self.repr

    def forward(self, x): # (T, B, C, *)
        if self.training:
            T, B, C = x.shape[:3]
            brdcst_shape = [T, 1, C]+[1] * (x.ndim - 3)
            mean = x.transpose(1, 2).mean(dim=list(range(2, x.ndim)), keepdim=True).transpose(1, 2) # (T, B, C, ...) -> (T, C, 1, ...*1) -> (T, 1, C, ...*1)
            var = x.transpose(1, 2).var(dim=list(range(2, x.ndim)), keepdim=True, unbiased=True).transpose(1, 2) # (T, B, C, ...) -> (T, C, 1, ...*1) -> (T, 1, C, ...*1)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x * self.gamma.view(brdcst_shape) + self.beta.view(brdcst_shape)
            self.running_mean = self.momentum * mean.view(T, C) + (1-self.momentum) * self.running_mean
            self.running_var = self.momentum * var.view(T, C) + (1-self.momentum) * self.running_var
        else:
            x = (x - self.running_mean.view(brdcst_shape)) / torch.sqrt(self.running_var.view(brdcst_shape) + self.eps) # transform the mean and var into (1, C, 1, 1) for broadcasting semantic
            x = x * self.gamma.view(brdcst_shape) + self.beta.view(brdcst_shape)
        return x


def check_bntt():
    T, B, C = 4, 10, 7
    x = torch.randn(T, B, C, 10, 20)
    bn = [BatchNorm(C) for _ in range(T)]
    bntt = BNTT(T, C)

    a = torch.stack([bn[i](x[i]) for i in range(x.shape[0])])
    b = bntt(x)

    print((a != b).sum())
check_bntt()



# %%
def check_mean():
    T, B, C, H, W = 4, 2, 3, 5, 5
    x = torch.arange(T*B*C*H*W).view(4, 2, 3, 5, 5).float()
    a = x.transpose(1, 2).mean(dim=list(range(2, x.ndim)), keepdim=True).transpose(1, 2) 
    b = reduce(x, 't b c h w -> t 1 c 1 1', 'mean')
    c = x.transpose(1, 2).var(dim=list(range(2, x.ndim)), keepdim=True, unbiased=True).transpose(1, 2) 
    d = reduce(x, 't b c h w -> t 1 c 1 1', torch.var) 
    return (a==b).sum().item() + (c == d).sum().item()