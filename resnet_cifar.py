# %%
import torch
from torch import nn
import torch.nn.functional as F

class BatchNorm(nn.Module):
    VARIANCE_UNBIASED = True # False will make it identical to nn's BatchNorm
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

class Block(nn.Module):
    CONV_BIAS = False
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=Block.CONV_BIAS),
                BatchNorm(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=Block.CONV_BIAS),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=Block.CONV_BIAS),
            BatchNorm(out_channels)
        )

    def forward(self, x):
        return F.relu(self.residual(x) + self.net(x))

        
class Storey(nn.Module):
    def __init__(self, depth, in_channels, out_channels, stride):
        super().__init__()

        net = [Block(in_channels, out_channels, stride=stride)]
        for _ in range(depth - 1):
            net.append(Block(out_channels, out_channels, stride=1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResNetCifar(nn.Module):
    LIN_BIAS = True
    CONV_BIAS = False
    def __init__(self, n=3, k=1):
        super().__init__()
        net = []
        net.append(nn.Conv2d(in_channels=3, out_channels=16*k, kernel_size=3, stride=1, padding=1, bias=self.CONV_BIAS))
        net.append(BatchNorm(16*k))
        net.append(nn.ReLU(inplace=True))
        
        net.append(Storey(depth=n, in_channels=16*k, out_channels=16*k, stride=2))
        net.append(Storey(depth=n, in_channels=16*k, out_channels=32*k, stride=2))
        net.append(Storey(depth=n, in_channels=32*k, out_channels=64*k, stride=2))

        self.net = nn.Sequential(
            *net,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64*k, 10, bias=self.LIN_BIAS),
        )
    def forward(self, x):
        return self.net(x)

count_layers = lambda mod: 1 if isinstance(mod, (nn.Conv2d, nn.Linear)) \
    else sum([count_layers(m) for n, m in mod.named_children() if n != 'residual'])
count_params = lambda model: sum([p.numel() for p in model.parameters()])

#%%
def get_data(ds, batch_size):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    if ds == 'CIFAR10':
        train_ds = datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))
        test_ds = datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    elif ds == 'CIFAR100':
        train_ds = datasets.CIFAR100(root="/tmp/data", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]))
        test_ds = datasets.CIFAR100(root="/tmp/data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, test_loader


class MixUp():
    def __init__(self, loss_fn, alpha):
        self.alpha = alpha
        self.loss_fn = loss_fn
    def __call__(self, x, y):
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()        
        shuffled = torch.randperm(y.shape[0])
        x_mix = lam * x + (1-lam) * x[shuffled]
        return x_mix, y, y[shuffled], lam
    
    def criterion(self, y, y_a, y_b, lam):
        return lam * self.loss_fn(y, y_a) + (1-lam) * self.loss_fn(y, y_b)
    
import math
class CutMix():
    def __init__(self, loss_fn, alpha):
        self.alpha = alpha
        self.loss_fn = loss_fn

    def randbbox(self, lam, H, W):
        c_rat = math.sqrt(1 - lam)
        h = H * c_rat
        w = W * c_rat
        c_h = torch.randint(0, H, (1,)).item()
        c_w = torch.randint(0, W, (1,)).item()
        x1, x2 = max(c_h - h // 2, 0), min(c_h + h // 2, H)
        y1, y2 = max(c_w - w // 2, 0), min(c_w + w // 2, W)
        return int(x1), int(x2), int(y1), int(y2)

    def __call__(self, x, y):
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x1, x2, y1, y2 = self.randbbox(lam, H, W)
        shuffled = torch.randperm(B)
        x_cut = x.clone()
        x_cut[:, :, x1:x2, y1:y2] = x[shuffled, :, x1:x2, y1:y2]
        true_lam = 1 - (x2-x1)*(y2-y1) / (H * W)
        return x_cut, y, y[shuffled], true_lam
    
    def criterion(self, y, y_a, y_b, lam):
        return lam * self.loss_fn(y, y_a) + (1-lam) * self.loss_fn(y, y_b)

def eval_model(model, loader, device, criterion):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total

def train(model, epochs, device, train_loader, test_loader, wandb_run):
    from torch import optim
    import random

    criterion = nn.CrossEntropyLoss()
    mixup = MixUp(loss_fn=criterion, alpha=0.2)
    cutmix = CutMix(loss_fn=criterion, alpha=1.0)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,   # epochs
        eta_min=0.0
    )
    def log_train_stat():
        count_explode = lambda m: sum([(p.grad.abs() > 1).sum().item() for p in m.parameters()])
        count_vanish = lambda m: sum([(p.grad.abs() < 1e-8).sum().item() for p in m.parameters()])
        explode = sum([count_explode(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))])
        vanish = sum([count_vanish(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))])
        wandb_run.log({
            'explode': explode, 
            'vanish': vanish
        })

    # Train
    count, freq = 0, 50
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mixup_or_cutmix = random.choice([mixup, cutmix])
            x, y_a, y_b, lam = mixup_or_cutmix(x, y)
            logits = model(x)
            loss = mixup_or_cutmix.criterion(logits, y_a, y_b, lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            count += 1
            if count % freq == 0 and wandb_run != None:
                log_train_stat()
        lr_scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        test_loss, test_acc = eval_model(model, test_loader, device, criterion)
        
        if wandb_run is not None:
            wandb_run.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_acc': test_acc,
            })
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

def get_best_gpu(min_required):
    import os
    max_arg = lambda a: (max(a), a.index(max(a)))
    free, gpu = max_arg(list(map(int, os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits').read().strip().split('\n'))))
    if free >= min_required:
        return gpu
    raise Exception("Not enough resources")

# %%
if __name__ == '__main__':
    device = f'cuda:{get_best_gpu(10000)}'
    ds = 'CIFAR10'
    batch_size=128
    train_loader, test_loader = get_data(ds, batch_size)
    model = ResNetCifar(n=18, k=2).to(device)
    run = None
    import wandb
    run = wandb.init(project='resnet')
    print(device)
    print(model)
    if run is not None:
        with open(__file__, 'r') as f:
            print(f.read())
        run.tags += (f"count_params={count_params(model):,}",)
        run.tags += (f"count_layers={count_layers(model)}",)
        run.tags += (f"batch_size={batch_size}",)
        run.tags += (f"ds={ds}",)
        run.tags += (f"augmented",)
    train(model, 200, device, train_loader, test_loader, wandb_run=run)
    run.finish()
