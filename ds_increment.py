from utils import BatchNorm
from custom_cifar100 import CIFAR100
import torch
from torch import nn

class VGG9(nn.Module):
    def __init__(self):
        C, H, W = 3, 32, 32
        super(VGG9, self).__init__()
        conv = [64, 'norm', 'act', 64, 'norm', 'act', 'pool',
            128, 'norm', 'act', 128, 'norm', 'act', 'pool',
            256, 'norm', 'act', 256, 'norm', 'act', 256, 'norm', 'act', 'pool',]
        fc = [1024, 'norm', 'act']
        net = []
        shrink = 1
        for l in conv:
            if isinstance(l, int):
                net.append(nn.Conv2d(C, l, kernel_size=3, padding=1, bias=False))
                C = l
            elif l == 'norm':
                net.append(BatchNorm(C))
            elif l == 'act':
                net.append(nn.ReLU(inplace=True))
            elif l == 'pool':
                net.append(nn.MaxPool2d(kernel_size=2, stride=2))
                shrink = shrink * 2
                
        net.append(nn.Flatten())
        C = C * (H // shrink) * (W // shrink)
        for l in fc:
            if isinstance(l, int):
                net.append(nn.Linear(C, l, bias=False))
                C = l
            if isinstance(l, float):
                net.append(nn.Dropout(l))
            elif l == 'norm':
                net.append(BatchNorm(C))
            elif l == 'act':
                net.append(nn.ReLU(inplace=True))
        self.pre_cls_size = C
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class CIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = VGG9()
        self.cls = nn.ModuleList([])

    def add_cls_head(self, num_cls):
        self.cls.append(nn.Linear(self.feature.pre_cls_size, num_cls))

    def prepare_for_task(self, task_no):
        for i, m in enumerate(self.cls):
            if i == task_no:
                m.requires_grad_(True)
            else:
                m.requires_grad_(False)

    def forward(self, x, task_no):
        out = self.feature(x)
        out = self.cls[task_no](out)
        return out

def get_data(dses, batch_size):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
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
            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

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
    # model.prepare_for_task(task['no'])

    from torch import optim
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
            total += len(y)

    print(f"Test acc task {task['name']}: {acc/total:.2f}")

if __name__ == "__main__":
    task_loaders = get_data(['CIFAR10', 'CIFAR100_coarse'], 128)
    device = 'cuda'
    model = CIL()
    import time
    for task in task_loaders:
        print(f"Training task: {task['name']}")
        start = time.time()
        train_task(model, task, device)
        print(f"Training time: {(time.time() - start):.2f}s")
        for i in range(task['no'] + 1):
            test_task(model, task_loaders[i], device)
