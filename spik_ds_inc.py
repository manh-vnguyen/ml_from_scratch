from spikingjelly.activation_based import neuron, layer, functional
from utils import BNTT, BatchNorm
import torch
from torch import nn

def reset_bntt(net):
    for m in net.modules():
        if isinstance(m, BNTT):
            m.reset()

class SVGG(nn.Module):
    conv_bias = False
    def __init__(self, C, H, W, T):
        super().__init__()
        conv = [64, 'bntt', 'act', 64, 'bntt', 'act', 'pool',
            128, 'bntt', 'act', 128, 'bntt', 'act', 'pool',
            256, 'bntt', 'act', 256, 'bntt', 'act', 256, 'bntt', 'act', 'pool',]
        feature = [
            # 2048, 'norm', 'act'
        ]
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
        for l in feature:
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
        self.pre_cls_size = C

    def forward(self, x):
        reset_bntt(self)
        out = x.repeat(self.T, 1, 1, 1, 1)
        for layer in self.net:
            out = layer(out)
            functional.reset_net(self.act)
            out = self.act(out)
        if self.last_layer is not None:
            out = self.last_layer(out)
        return out

class CIL(nn.Module):
    def __init__(self, C, H, W, T):
        super().__init__()
        self.feature = SVGG(C, H, W, T)
        self.cls = nn.ModuleList([])

    def add_cls_head(self, num_cls):
        self.cls.append(layer.MultiStepContainer(
            nn.Linear(self.feature.pre_cls_size, num_cls)
        ))

    def forward(self, x, task_no):
        out = self.feature(x)
        out = self.cls[task_no](out)
        return torch.mean(out, dim=0)

def get_data(dses, batch_size):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from custom_cifar100 import CIFAR100
    task_loaders = []
    for ds in dses:
        if ds == 'CIFAR10':
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_ds = datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=T)
            test_ds = datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=T)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        elif ds.startswith('CIFAR100'):
            assert ds in ['CIFAR100_coarse', 'CIFAR100_fine'], "Can't recognize task"
            coarse_label = True if ds == 'CIFAR100_coarse' else False
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            train_ds = CIFAR100(root="/tmp/data", train=True, coarse_label=coarse_label, download=True, transform=T)
            test_ds = CIFAR100(root="/tmp/data", train=False, coarse_label=coarse_label, download=True, transform=T)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        task_loaders.append({
            'name': ds,
            'no': len(task_loaders),
            'train': train_loader,
            'test': test_loader,
            'num_cls': len(train_ds.classes)
        })
    return task_loaders

#%%
def train_task(model, task, device):
    model.add_cls_head(task['num_cls'])
    model = model.to(device)

    from torch import optim
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    for _ in range(3):
        for x, y in task['train']:
            x, y = x.to(device), y.to(device)
            logits = model(x, task['no'])
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test_task(model, task, device):
    acc, total = 0, 0
    with torch.no_grad():
        for x, y in task['test']:
            x, y = x.to(device), y.to(device)
            logits = model(x, task['no'])
            acc += torch.sum(torch.argmax(logits, dim=1) == y).item()
            total += y.shape[0]

    print(f"Test acc task {task['name']}: {acc/total:.2f}")

if __name__ == "__main__":
    task_loaders = get_data(['CIFAR10', 'CIFAR100_coarse'], 128)
    device = 'cuda'
    model = CIL(C=3, H=32, W=32, T=4)
    import time
    for task in task_loaders:
        print(f"Training task: {task['name']}")
        start = time.time()
        train_task(model, task, device)
        print(f"Training time: {(time.time() - start):.2f}s")
        for i in range(task['no'] + 1):
            test_task(model, task_loaders[i], device)
