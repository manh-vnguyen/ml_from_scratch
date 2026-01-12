# %%
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    conv_bias = False
    def __init__(self, in_channels, out_channels, stride, residual):
        super().__init__()
        if residual:
            if in_channels != out_channels or stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=Block.conv_bias),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.residual = nn.Identity()
        else:
            self.residual = None
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=Block.conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=Block.conv_bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.residual is not None:
            out = self.residual(x) + self.net(x)
        else:
            out = self.net(x)
        return F.relu(out)

        
class Storey(nn.Module):
    def __init__(self, depth, in_channels, out_channels, stride, residual):
        super().__init__()

        net = [Block(in_channels, out_channels, stride=stride, residual=residual)]
        for _ in range(depth - 1):
            net.append(Block(out_channels, out_channels, stride=1, residual=residual))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    lin_bias = True
    def __init__(self, C, H, W, convfig=[3, 4, 6, 3], conv_bias=False, residual=False):
        super().__init__()
        net = []
        net.append(nn.Conv2d(in_channels=C, out_channels=64, kernel_size=3, stride=1, padding=1, bias=conv_bias))
        net.append(nn.BatchNorm2d(64))
        net.append(nn.ReLU(inplace=True))
        # net.append(nn.MaxPool2d(2, 2))

        Block.conv_bias = conv_bias
        net.append(Storey(depth=convfig[0], in_channels=64, out_channels=64, stride=1, residual=residual))
        net.append(Storey(depth=convfig[1], in_channels=64, out_channels=128, stride=2, residual=residual))
        net.append(Storey(depth=convfig[2], in_channels=128, out_channels=256, stride=2, residual=residual))
        net.append(Storey(depth=convfig[3], in_channels=256, out_channels=512, stride=2, residual=residual))

        shrink = 2**3        
        self.net = nn.Sequential(
            *net,
            nn.Flatten(),
            nn.Linear(512 * (H//shrink) * (W//shrink), 10, bias=ResNet.lin_bias),
        )
    def forward(self, x):
        return self.net(x)
    
# %%
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
BATCH_SIZE = 128
train_ds = datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# %%

def epoch_stat(model):
    count_explode = lambda m: sum([(p.grad.abs() > 1).sum().item() for p in m.parameters()])
    count_vanish = lambda m: sum([(p.grad.abs() < 1e-8).sum().item() for p in m.parameters()])
    explode = [count_explode(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    vanish = [count_vanish(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    return explode, vanish

def model_stat(model):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    x, y = next(iter(train_loader))
    print(type(model), sum(p.numel() for p in model.parameters()))
    loss = loss_fn(model(x), y)
    model.zero_grad()
    loss.backward()
    explode, vanish = epoch_stat(model)
    print("Exploding:", explode)
    print("Vanishing:", vanish)

def collect_stats():
    from vgg import vgg9, vgg19
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    device = 'cuda'
    vgg = vgg19(10, 3, 32, 32).to(device)
    resnet = ResNet(3, 32, 32, convfig=[6, 6, 6, 6], conv_bias=False).to(device)


    v_opt = torch.optim.AdamW(vgg.parameters(), lr=1e-3)
    r_opt = torch.optim.AdamW(resnet.parameters(), lr=1e-3)

    v_stats = []
    r_stats = []
    train_iter = iter(train_loader)
    for x, y in train_iter:
        x, y = x.to(device), y.to(device)
        v_loss = loss_fn(vgg(x), y)
        r_loss = loss_fn(resnet(x), y)
        v_opt.zero_grad()
        r_opt.zero_grad()
        v_loss.backward()
        r_loss.backward()
        v_stats.append((v_loss.item(), epoch_stat(vgg)))
        r_stats.append((r_loss.item(), epoch_stat(resnet)))
        torch.nn.utils.clip_grad_norm_(vgg.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=5.0)
        v_opt.step()
        r_opt.step()

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    v_vanish = np.array([i[1][0] for i in v_stats]).T
    v_explode = np.array([i[1][1] for i in v_stats]).T
    r_vanish = np.array([i[1][0] for i in r_stats]).T
    r_explode = np.array([i[1][1] for i in r_stats]).T

    print(r_vanish.max())
    print(r_explode.max())
    print(v_vanish.max())
    print(v_explode.max())

    sns.heatmap(v_explode, cmap='RdYlBu')
    plt.show()
    sns.heatmap(v_vanish, cmap='RdYlBu')
    plt.show()
    sns.heatmap(r_explode, cmap='RdYlBu')
    plt.show()
    sns.heatmap(r_vanish, cmap='RdYlBu')
    plt.show()
# %%

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

def train(model, epochs, device, wandb_run):
    # Model, loss, optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    param_groups = get_parameter_groups(model, weight_decay=1e-2)
    optimizer = torch.optim.AdamW(param_groups, lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=32, eta_min=1e-8)

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


def count_params(model):
    count = 0
    for n, p in model.named_parameters():
        # print(n, p.shape)
        count += p.numel()
    return count

# %%
if __name__ == '__main__':
    import wandb
    device = 'cuda'
    model = ResNet(3, 32, 32, convfig=[3, 4, 6, 3], conv_bias=False, residual=True).to(device)
    total_params = count_params(model)
    run = wandb.init(project='resnet', 
                     notes="ResNet(3, 32, 32, convfig=[3, 4, 6, 3], conv_bias=False, residual=True) - rearrange ReLU"
                    )
    print(model)
    print(total_params)
    if run is not None:
        run.tags += (f"count_params={count_params(model)}",)
        run.tags += (f"batch_size={BATCH_SIZE}",)
    train(model, 100,'cuda', wandb_run=run)
    run.finish()