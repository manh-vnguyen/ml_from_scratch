# %%
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



class VGG(nn.Module):
    def __init__(self, conv, cls, C, H, W):
        super(VGG, self).__init__()
        net = []
        shrink = 1
        for l in conv:
            if isinstance(l, int):
                net.append(nn.Conv2d(C, l, kernel_size=3, padding=1, bias=False))
                C = l
            elif l == 'norm':
                net.append(nn.BatchNorm2d(C))
            elif l == 'act':
                net.append(nn.ReLU(inplace=True))
            elif l == 'pool':
                net.append(nn.MaxPool2d(kernel_size=2, stride=2))
                shrink = shrink * 2
                

        
        net.append(nn.Flatten())
        C = C * (H // shrink) * (W // shrink)
        for l in cls:
            if isinstance(l, int):
                net.append(nn.Linear(C, l, bias=False))
                C = l
            elif l == 'norm':
                net.append(nn.BatchNorm1d(C))
            elif l == 'act':
                net.append(nn.ReLU(inplace=True))
                

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
    
def vgg9(num_classes, C, H, W):
    conv = [64, 'norm', 'act', 64, 'norm', 'act', 'pool',
            128, 'norm', 'act', 128, 'norm', 'act', 'pool',
            256, 'norm', 'act', 256, 'norm', 'act', 256, 'norm', 'act', 'pool',]
    cls = [1024, 'norm', 'act', num_classes]
    return VGG(conv, cls, C, H, W)

def vgg19(num_classes, C, H, W):
    conv = [64, 'norm', 'act', 64, 'norm', 'act', 'pool',
            128, 'norm', 'act', 128, 'norm', 'act', 'pool',
            256, 'norm', 'act', 256, 'norm', 'act', 256, 'norm', 'act', 256, 'norm', 'act', 'pool',
            512, 'norm', 'act', 512, 'norm', 'act', 512, 'norm', 'act', 512, 'norm', 'act', 'pool',
            512, 'norm', 'act', 512, 'norm', 'act', 512, 'norm', 'act', 512, 'norm', 'act', 'pool',]
    cls = [4096, 'norm', 'act', 4096, 'norm', 'act', num_classes]
    return VGG(conv, cls, C, H, W)

# %% 
def count_params(model):
    count = 0
    for n, p in model.named_parameters():
        # print(n, p.shape)
        count += p.numel()
    return count


BATCH_SIZE = 128

def get_data(ds):
    if ds == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        
        train_ds = datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    elif ds == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        train_ds = datasets.CIFAR100(root="/tmp/data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR100(root="/tmp/data", train=False, download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, test_loader

#%%
def get_parameter_groups(model, weight_decay):
    """
    Separate parameters into two groups: those that should have weight decay
    and those that shouldn't (LayerNorm parameters: gamma and beta).
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if this is a LayerNorm parameter (gamma or beta)
            if any(ln_param in name for ln_param in ['gamma', 'beta']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

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
    # Model, loss, optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # param_groups = get_parameter_groups(model, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=32, eta_min=1e-6)

    # Train
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            running_loss += loss.item() * y.size(0)

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
    ds = 'CIFAR100'
    train_loader, test_loader = get_data(ds)
    model = vgg19(num_classes=100, C=3, H=32, W=32).to(device)
    run = wandb.init(project='VGG', 
                     notes="vgg19(num_classes=100, C=3, H=32, W=32)"
                    )
    print(model)
    if run is not None:
        run.tags += (f"count_params={count_params(model)}",)
        run.tags += (f"batch_size={BATCH_SIZE}",)
        run.tags += (f"ds={ds}",)
    train(model, 100,'cuda', train_loader, test_loader, wandb_run=run)
    run.finish()

__all__ = ['vgg9', 'vgg19']