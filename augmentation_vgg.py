#%%
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import optim
from torch import nn


def count_params(model):
    count = 0
    for n, p in model.named_parameters():
        # print(n, p.shape)
        count += p.numel()
    return count

BATCH_SIZE = 256

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
#%%
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
def get_parameter_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
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
    import random

    criterion = nn.CrossEntropyLoss()
    mixup = MixUp(loss_fn=criterion, alpha=0.2)
    cutmix = CutMix(loss_fn=criterion, alpha=1.0)
    
    # param_groups = get_parameter_groups(model, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader)*10, eta_min=1e-6
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
    best_test_acc = 0
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
            lr_scheduler.step()
            running_loss += loss.item() * y.size(0)
            count += 1
            if count % freq == 0 and wandb_run != None:
                log_train_stat()

        train_loss = running_loss / len(train_loader.dataset)
        test_loss, test_acc = eval_model(model, test_loader, device, criterion)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            torch.save({
                'model': model.state_dict(), 
                'test_acc': test_acc
            }, 'best_vgg_cifar10.pt')
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
    from vgg import vgg9, vgg19
    import wandb
    device = f'cuda:{get_best_gpu(10000)}'
    ds = 'CIFAR10'
    train_loader, test_loader = get_data(ds)
    model = vgg9(num_classes=10, C=3, H=32, W=32).to(device)
    run = None
    run = wandb.init(project='VGG',
                     notes="")
    print(model)
    if run is not None:
        run.save(__file__)
        run.tags += (f"count_params={count_params(model)}",)
        run.tags += (f"batch_size={BATCH_SIZE}",)
        run.tags += (f"ds={ds}",)
        run.tags += (f"augmented",)
    train(model, 1000, device, train_loader, test_loader, wandb_run=run)
    run.finish()

