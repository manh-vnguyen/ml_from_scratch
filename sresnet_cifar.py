#%%
from torch import nn
import torch
from spikingjelly.activation_based import functional, neuron, layer, surrogate
from utils import BatchNorm, BNTT, get_best_gpu
import torch.nn.functional as F
from torch.utils.data import Subset
from ds import get_ds
from fedlib import eval_model
import numpy as np

def conv1x1(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def spiking_act():
    # return neuron.LIFNode(
    #         tau=3.0,
    #         v_threshold=1.0,
    #         v_reset=0.0,
    #         surrogate_function=surrogate.ATan(),
    #         detach_reset=True,
    #         step_mode='s'
    #     )
    return neuron.IFNode(
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode='s'
        )

def norm(T, C):
    # return BatchNorm(C)
    return BNTT(T, C)

def block(in_channels, out_channels, stride, T):
    # return SBlock(in_channels, out_channels, stride, T)
    return SEWBlock(in_channels, out_channels, stride, T, 'IAND')

class SBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, T):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                # norm(T, out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.net = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            norm(T, out_channels),
            spiking_act(),
            conv3x3(out_channels, out_channels, stride=1),
            norm(T, out_channels),
        )

        self.act = spiking_act()

    def forward(self, x):
        return self.act(self.residual(x) + self.net(x))


class SEWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, T):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                norm(T, out_channels),
                spiking_act(),
            )
        else:
            self.residual = nn.Identity()

        self.net = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            norm(T, out_channels),
            spiking_act(),
            conv3x3(out_channels, out_channels, stride=1),
            norm(T, out_channels),
            spiking_act(),
        )

class SEWBlockADD(SEWBlock):
    def forward(self, x):
        return self.residual(x) + self.net(x)
    
class SEWBlockAND(SEWBlock):
    def forward(self, x):
        return self.residual(x) * self.net(x)

class SEWBlockIAND(SEWBlock):
    def forward(self, x):
        return self.residual(x) * (1 - self.net(x))
        
class SStorey(nn.Module):
    def __init__(self, block, depth, in_channels, out_channels, stride, T):
        super().__init__()

        net = [block(in_channels, out_channels, stride, T)]
        for _ in range(depth - 1):
            net.append(block(out_channels, out_channels, stride=1, T=T))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class SResNetCifar(nn.Module):
    def __init__(self, block, n, k, T):
        super().__init__()
        self.T = T

        self.net = nn.Sequential(
            conv3x3(in_channels=3, out_channels=16*k, stride=1),
            norm(T, 16*k),
            spiking_act(),
            SStorey(block=block, depth=n, in_channels=16*k, out_channels=16*k, stride=2, T=T),
            SStorey(block=block, depth=n, in_channels=16*k, out_channels=32*k, stride=2, T=T),
            SStorey(block=block, depth=n, in_channels=32*k, out_channels=64*k, stride=2, T=T),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64*k, 1024, bias=True),
            norm(T, 1024),
            spiking_act(),
            nn.Linear(1024, 10, bias=True),
        )

    def reset_net(self):
        for m in self.modules():
            if hasattr(m, 'reset'):
                m.reset()

    def forward(self, x):
        self.reset_net()
        out = 0
        for _ in range(self.T):
            out = out + self.net(x)
        out = out / self.T
        return out
    

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernels):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                BatchNorm(out_channels),
            )
        else:
            self.residual = nn.Identity()

        layers = []
        for i, k in enumerate(kernels):
            if i == 0: c, s = in_channels, stride 
            else: c, s = out_channels, 1
            conv = conv1x1 if k == 1 else conv3x3
            layers += [conv(c, out_channels, s), BatchNorm(out_channels)]
            if i != len(kernels) - 1:
                layers += [nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.relu(self.residual(x) + self.net(x))


class B131(Block):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(in_channels, out_channels, stride, kernels=[1, 3, 1])

class B31(Block):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(in_channels, out_channels, stride, kernels=[3, 1])

class B13(Block):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(in_channels, out_channels, stride, kernels=[1, 3])

class B311(Block):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(in_channels, out_channels, stride, kernels=[3, 1, 1])

class B33(Block):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(in_channels, out_channels, stride, kernels=[3, 3])

class B313(Block):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__(in_channels, out_channels, stride, kernels=[3, 1, 3])
        
class Storey(nn.Module):
    def __init__(self, block, depth, in_channels, out_channels, stride):
        super().__init__()

        net = [block(in_channels, out_channels, stride=stride)]
        for _ in range(depth - 1):
            net.append(block(out_channels, out_channels, stride=1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResNetCifar(nn.Module):
    def __init__(self, block, n, k):
        super().__init__()
        net = []
        net.append(conv3x3(in_channels=3, out_channels=16*k, stride=1))
        net.append(BatchNorm(16*k))
        net.append(nn.ReLU(inplace=True))
        
        net.append(Storey(block=block, depth=n, in_channels=16*k, out_channels=16*k, stride=2))
        net.append(Storey(block=block, depth=n, in_channels=16*k, out_channels=32*k, stride=2))
        net.append(Storey(block=block, depth=n, in_channels=32*k, out_channels=64*k, stride=2))

        self.net = nn.Sequential(
            *net,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64*k, 1024, bias=True),
            BatchNorm(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10, bias=True),
        )
    def forward(self, x):
        return self.net(x)

count_layers = lambda mod: 1 if isinstance(mod, (nn.Conv2d, nn.Linear)) \
    else sum([count_layers(m) for n, m in mod.named_children() if n != 'residual'])

count_layers_wrn = lambda mod: 1 if isinstance(mod, (nn.Conv2d)) \
    else sum([count_layers_wrn(m) for n, m in mod.named_children()])

count_params = lambda model: sum([p.numel() for p in model.parameters()])

def log_train_stat(model):
    count_explode = lambda m: sum([(p.grad.abs() > 1).sum().item() for p in m.parameters()])
    count_vanish = lambda m: sum([(p.grad.abs() < 1e-8).sum().item() for p in m.parameters()])
    explode = sum([count_explode(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))])
    vanish = sum([count_vanish(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))])
    wandb_run.log({
        'explode': explode, 
        'vanish': vanish
    })


def train(model, opt, train_loader, test_loader, device):
    print(f"Model: {type(model)}, Number of layers: {count_layers_wrn(model)}, number of params: {count_params(model)}")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x) 
            loss = criterion(logits, y)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss = total_loss/len(train_loader)
        test_loss, test_acc = eval_model(model, test_loader, device, nn.CrossEntropyLoss())
        print(f"{epoch=}, {train_loss=}, {test_loss=}, {test_acc=}")
        if wandb_run is not None:
            wandb_run.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_acc': test_acc,
            })

#%%
if __name__ == '__main__':
    import wandb
    wandb_run = wandb.init(project='sresnet')
    # if wandb_run is not None:
    #     with open(__file__, 'r') as f:
    #         print(f.read())

    train_ds, test_ds = get_ds('CIFAR10')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128)

    device = f'cuda:{get_best_gpu(1100)}'

    SEW_BLOCKS = [SEWBlockADD]

    for block_cls in SEW_BLOCKS:
        print(f"\n=== Training with block: {block_cls.__name__} ===")
        model = SResNetCifar(block=block_cls, n=6, k=2, T=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        train(model, opt, train_loader, test_loader, device)

    # BLOCKS = [B33]
    # for block_cls in BLOCKS:
    #     print(f"\n=== Training with block: {block_cls.__name__} ===")
    #     model = ResNetCifar(block=block_cls, n=6, k=2).to(device)
    #     opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    #     train(model, opt, train_loader, test_loader, device)
