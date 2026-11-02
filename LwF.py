#%%
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
        self.feature_size = C
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class CIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_net = VGG9()
        self.task_head = nn.ModuleList([])

    def add_task_head(self, num_cls):
        self.task_head.append(nn.Linear(self.shared_net.feature_size, num_cls))

    def forward(self, x, task_no):
        out = self.shared_net(x)
        out = self.task_head[task_no](out)
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
"""
model = CIL()
model.add_task_head(10)
opt = torch.optim.AdamW(model.parameters())
x = torch.randn(5, 3, 32, 32)
y = torch.randint(0, 10, (5,))
from copy import deepcopy
frozen_model = deepcopy(model)
frozen_model.requires_grad_(False)
logits = model(x, 0)
print(logits)
print(frozen_model(x, 0))
loss = torch.nn.functional.cross_entropy(logits, y)
opt.zero_grad()
loss.backward()
opt.step()
print(model(x, 0))
"""


#%%

#%% 
import torch
import torch.nn.functional as F

def loss_new(y_hat, y):
    # y_one_hot = F.one_hot(y, num_cls)
    # y_hat_softmax = F.softmax(y_hat, 0)
    # return torch.sum(-y * torch.log(y_hat_softmax))
    return F.cross_entropy(y_hat, y)

def loss_old(y_hat, y, T):
    modif_y = lambda y: y ** (1/T) / torch.sum(y ** (1/T))
    return torch.sum(-modif_y(y) * torch.log(modif_y(y_hat)))

#%%
def train_task(model, task, device):
    from torch import optim
    from copy import deepcopy
    
    if task['no'] == 1:
        frozen_model = deepcopy(model)
        frozen_model.requires_grad_(False)
        criterion = loss_new
    else:
        frozen_model = None
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.add_task_head(task['num_cls'])
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    for _ in range(3):
        for x, y in task['train']:
            x, y = x.to(device), y.to(device)
            logits = model(x, task['no'])
            loss = criterion(logits, y)
            if frozen_model is not None:
                y_old = frozen_model(x, task['no'] - 1)
                y_hat_old = model(x, task['no'] - 1)
                loss = loss + 0.1 * loss_old(y_hat_old, y_old, T = 1)

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

    print(f"Test acc task {task['name']}: {acc/total*100:.2f}")

if __name__ == "__main__":
    task_loaders = get_data(['CIFAR10', 
                             'CIFAR100_coarse'
                             ], 128)
    device = 'cuda'
    model = CIL()
    model.add_task_head(10)
    model.load_state_dict(torch.load('vgg_cifar10.pth', map_location='cpu'))
    model = model.to(device)
    test_task(model, task_loaders[0], device)
    for task in task_loaders[1:]:
        print(f"Training task: {task['name']}")
        train_task(model, task, device)
        for i in range(task['no'] + 1):
            test_task(model, task_loaders[i], device)
