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
# transform = transforms.Compose([
#             transforms.ToTensor()
#         ])
        
# train_ds = datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=transform)
# test_ds = datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=transform)
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
# x, y = next(iter(train_loader))

#%%
class CutMix():
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

# cutmix = CutMix(nn.CrossEntropyLoss(), alpha=1.0)
# x_mix, y_a, y_b, lam = cutmix(x, y)
# # %%
# import matplotlib.pyplot as plt
# plt.imshow(x_mix[0].permute(1,2,0).numpy())
# plt.show()
# plt.imshow(x_mix[0].permute(1,2,0).numpy())
# plt.show()
# plt.imshow(x_mix[0].permute(1,2,0).numpy())
# plt.show()

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
    # Model, loss, optimizer
    criterion = nn.CrossEntropyLoss()
    cutmix = CutMix(loss_fn=criterion, alpha=1.0)
    # param_groups = get_parameter_groups(model, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, eta_min=1e-6)

    # Train
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if cutmix is not None:
                x, y_a, y_b, lam = cutmix(x, y)
            logits = model(x)
            loss = cutmix.criterion(logits, y_a, y_b, lam)
            optimizer.zero_grad()
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
    from vgg import vgg9
    import wandb
    device = 'cuda'
    ds = 'CIFAR10'
    train_loader, test_loader = get_data(ds)
    model = vgg9(num_classes=10, C=3, H=32, W=32).to(device)
    run = None
    run = wandb.init(project='VGG', 
                     notes="vgg9(num_classes=10, C=3, H=32, W=32)"
                    )
    print(model)
    if run is not None:
        run.tags += (f"count_params={count_params(model)}",)
        run.tags += (f"batch_size={BATCH_SIZE}",)
        run.tags += (f"ds={ds}",)
        run.tags += (f"cutmix",)
    train(model, 100,'cuda', train_loader, test_loader, wandb_run=run)
    run.finish()
