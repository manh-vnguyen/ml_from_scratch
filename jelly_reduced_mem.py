#%%
import torch
from torch import nn
from spikingjelly.activation_based import functional, neuron, layer

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
    
def reset_bntt(net):
    for m in net.modules():
        if isinstance(m, BNTT):
            m.reset()

class SVGG_REDUCED_MEM(nn.Module):
    conv_bias = False
    def __init__(self, num_classes, C, H, W, T):
        super().__init__()
        conv = [64, 'bntt', 'act', 64, 'bntt', 'act', 'pool',
            128, 'bntt', 'act', 128, 'bntt', 'act', 'pool',
            256, 'bntt', 'act', 256, 'bntt', 'act', 256, 'bntt', 'act', 'pool',]
        cls = [1024, 'norm', 'act', num_classes]
        self.act = neuron.IFNode(step_mode='m', v_reset=None)
        self.T = T
        net = []
        shrink = 1
        block = []
        for l in conv:
            if isinstance(l, int):
                block.append(nn.Conv2d(in_channels=C, out_channels=l, kernel_size=3, stride=1, padding=1, bias=self.conv_bias))
                C = l
            elif l == 'norm':
                block.append(BatchNorm(C))
            elif l == 'bntt':
                block.append(BNTT(T, C))
            elif l == 'act':
                if len(block) > 0: net.append(layer.MultiStepContainer(*block))
                block = []
            elif l == 'pool':
                block.append(nn.MaxPool2d(2, 2))
                shrink *= 2
        C = C * H * W // shrink**2
        block.append(nn.Flatten())
        for l in cls:
            if isinstance(l, int):
                block.append(nn.Linear(C, l))
                C = l
            elif l == 'norm':
                block.append(BatchNorm(C))
            elif l == 'bntt':
                block.append(BNTT(T, C))
            elif l == 'act':
                if len(block) > 0: net.append(layer.MultiStepContainer(*block))
                block = []

        self.net = nn.ModuleList(net)
        if len(block) > 0: 
            self.last_layer = layer.MultiStepContainer(*block) # This last layer have no activation
        else:
            self.last_layer = None

    def forward(self, x):
        reset_bntt(self)
        out = x.repeat(self.T, 1, 1, 1, 1)
        for layer in self.net:
            out = layer(out)
            functional.reset_net(self.act)
            out = self.act(out)
        if self.last_layer is not None:
            out = self.last_layer(out)
        return torch.mean(out, dim=0)


#%%
BATCH_SIZE = 128

def get_data(ds):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    if ds == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_ds = datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    elif ds == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        train_ds = datasets.CIFAR100(root="/tmp/data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR100(root="/tmp/data", train=False, download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

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
    from torch import optim
    # Model, loss, optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=32, eta_min=1e-6)

    def log_train_stat():
        count_explode = lambda m: sum([(p.grad.abs() > 1).sum().item() for p in m.parameters()])
        count_vanish = lambda m: sum([(p.grad.abs() < 1e-8).sum().item() for p in m.parameters()])
        explode = sum([count_explode(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))])
        vanish = sum([count_vanish(m) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))])
        wandb_run.log({
            'explode': explode, 
            'vanish': vanish
        })

    count, freq = 0, 50
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

            count += 1
            if count % freq == 0 and wandb_run != None:
                log_train_stat()

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
    from utils import get_best_gpu
    import wandb
    device = f'cuda:{get_best_gpu(6000)}'
    print(device)
    ds = 'CIFAR10'
    train_loader, test_loader = get_data(ds)
    model = SVGG_REDUCED_MEM(num_classes=10, C=3, H=32, W=32, T=4).to(device)
    run = None
    run = wandb.init(project='SVGG', notes='Add add one fully connected layer')
    print(model)
    if run is not None:
        with open(__file__, 'r') as f:
            print(f.read())
        run.tags += (f"count_params={sum([p.numel() for p in model.parameters()])}",)
        run.tags += (f"batch_size={BATCH_SIZE}",)
        run.tags += (f"ds={ds}",)
        run.tags += (f"reduced_mem",)
        run.tags += (f"no_augment",)
    train(model, 100, device, train_loader, test_loader, wandb_run=run)
    run.finish()
