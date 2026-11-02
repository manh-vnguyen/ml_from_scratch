#%%
import torch
from torch import nn
from spikingjelly.activation_based import neuron, functional, layer

def check_IF():
    T, B, C, H, W = 100, 300, 3, 5, 5
    x = torch.rand(T, B, C, H, W)
    net = neuron.IFNode(step_mode='m', v_reset=None)
    membrane = torch.zeros(*x.shape[1:])
    spikes = []
    for i in x:
        membrane += i
        spike_map = torch.zeros(*x.shape[1:])
        spike_map[membrane > 1.] = 1.
        membrane[membrane>1.] = membrane[membrane>1.] - 1.
        spikes.append(spike_map)
    spikes = torch.stack(spikes)
    single = neuron.IFNode(step_mode='s', v_reset=None)
    IF_spikes = []
    for i in x:
        IF_spikes.append(single(i))
    IF_spikes = torch.stack(IF_spikes)
    print(torch.sum(spikes != net(x)), (IF_spikes != spikes).sum())

def check_multistep_module():
    T, B, C, H, W = 25, 32, 3, 5, 5
    x = torch.rand(T, B, C, H, W)
    conv = nn.Conv2d(3, 2, 3, 1, 1)
    bn = nn.BatchNorm2d(2)
    (bn(conv(x[0])) != functional.multi_step_forward(x, (conv, bn))[0]).sum()
    net = layer.MultiStepContainer(conv, bn)
    (bn(conv(x[-2])) != net(x)[-2]).sum()

class BatchNorm(nn.Module):
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
            var = x.transpose(0, 1).var(dim=list(range(1, x.ndim)), keepdim=True, unbiased=True).transpose(0, 1) # (1, C, 1, 1)
            x = (x - mean) / torch.sqrt(var + self.eps) # transform the mean and var into (1, C, 1, 1) for broadcasting semanticj
            x = x * self.gamma.view([-1]+[1] * (x.ndim - 2)) + self.beta.view([-1]+[1] * (x.ndim - 2))
            self.running_mean = self.momentum * mean.view(-1) + (1-self.momentum) * self.running_mean
            self.running_var = self.momentum * var.view(-1) + (1-self.momentum) * self.running_var
        else:
            x = (x - self.running_mean.view([-1]+[1] * (x.ndim - 2))) / torch.sqrt(self.running_var.view([-1]+[1] * (x.ndim - 2)) + self.eps) # transform the mean and var into (1, C, 1, 1) for broadcasting semantic
            x = x * self.gamma.view([-1]+[1] * (x.ndim - 2)) + self.beta.view([-1]+[1] * (x.ndim - 2))
        return x

#%%
class BNTT(nn.Module):
    def __init__(self, num_channels, T):
        super().__init__()
        self.bntt = nn.ModuleList([BatchNorm(num_channels) for _ in range(T)])
        self.t = 0

    def reset(self):
        self.t = 0

    def forward(self, x):
        out = self.bntt[self.t](x)
        self.t += 1
        return out

#%%
class SVGG(nn.Module):
    conv_bias = False
    def __init__(self, num_classes, C, H, W, T):
        super().__init__()
        conv = [64, 'norm', 'act', 64, 'norm', 'act', 'pool',
            128, 'norm', 'act', 128, 'norm', 'act', 'pool',
            256, 'norm', 'act', 256, 'norm', 'act', 256, 'norm', 'act', 'pool',]
        cls = [num_classes]
        self.T = T
        net = []
        shrink = 1
        for l in conv:
            if isinstance(l, int):
                net.append(layer.MultiStepContainer(
                    nn.Conv2d(in_channels=C, out_channels=l, kernel_size=3, stride=1, padding=1, bias=self.conv_bias)
                ))
                C = l
            elif l == 'norm':
                net.append(layer.MultiStepContainer(
                    BatchNorm(C)
                ))
            elif l == 'act':
                net.append(neuron.IFNode(v_reset=None, step_mode='m'))
            elif l == 'pool':
                net.append(layer.MultiStepContainer(
                    nn.MaxPool2d(2, 2)
                ))
                shrink *= 2

        C = C*H*W // shrink**2
        net.append(nn.Flatten(start_dim=2))
        for l in cls:
            if isinstance(l, int):
                net.append(layer.MultiStepContainer(
                    nn.Linear(C, l)
                ))
                C = l
            elif l == 'norm':
                net.append(layer.MultiStepContainer(
                    BatchNorm(C)
                ))
            elif l == 'act':
                net.append(neuron.IFNode(v_reset=None, step_mode='m'))

        self.net = nn.Sequential(*net)
    def forward(self, x):
        functional.reset_net(self)
        x = x.repeat(self.T, 1, 1, 1, 1)
        out = self.net(x)
        return torch.mean(out, dim=0)

def reset_net(net):
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()

class SVGG_SINGLE_STEP(nn.Module):
    conv_bias = False
    def __init__(self, num_classes, C, H, W, T):
        super().__init__()
        conv = [64, 'bntt', 'act', 64, 'bntt', 'act', 'pool',
            128, 'bntt', 'act', 128, 'bntt', 'act', 'pool',
            256, 'bntt', 'act', 256, 'bntt', 'act', 256, 'bntt', 'act', 'pool',]
        cls = [num_classes]
        self.T = T
        net = []
        shrink = 1
        for l in conv:
            if isinstance(l, int):
                net.append(nn.Conv2d(in_channels=C, out_channels=l, kernel_size=3, stride=1, padding=1, bias=self.conv_bias))
                C = l
            elif l == 'norm':
                net.append(BatchNorm(C))
            elif l == 'bntt':
                net.append(BNTT(C, T))
            elif l == 'act':
                net.append(neuron.IFNode(step_mode='s', v_reset=None))
            elif l == 'pool':
                net.append(nn.MaxPool2d(2, 2))
                shrink *= 2
        C = C * H * W // shrink**2
        net.append(nn.Flatten())
        for l in cls:
            if isinstance(l, int):
                net.append(nn.Linear(C, l))
                C = l
            elif l == 'bntt':
                net.append(BNTT(C, T))
            elif l == 'norm':
                net.append(BatchNorm(C))
            elif l == 'act':
                net.append(neuron.IFNode(step_mode='s', v_reset=None))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        reset_net(self)
        out = 0
        for _ in range(self.T):
            out = out + self.net(x)
        out = out / self.T
        return out

#%%
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
BATCH_SIZE = 128

def get_data(ds):
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
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
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
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
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

#%%
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
    import random

    criterion = nn.CrossEntropyLoss()
    mixup = MixUp(loss_fn=criterion, alpha=0.2)
    cutmix = CutMix(loss_fn=criterion, alpha=1.0)
    
    # param_groups = get_parameter_groups(model, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=1e-6
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

# %%
if __name__ == '__main__':
    import wandb
    device = 'cuda'
    ds = 'CIFAR10'
    train_loader, test_loader = get_data(ds)
    model = SVGG_SINGLE_STEP(num_classes=10, C=3, H=32, W=32, T=10).to(device)
    run = None
    # run = wandb.init(project='SVGG')
    print(model)
    if run is not None:
        run.save(__file__)
        run.tags += (f"count_params={sum([p.numel() for p in model.parameters()])}",)
        run.tags += (f"batch_size={BATCH_SIZE}",)
        run.tags += (f"ds={ds}",)
        run.tags += (f"augmented",)
    train(model, 100, device, train_loader, test_loader, wandb_run=run)
    run.finish()


__all__ = ['SVGG']