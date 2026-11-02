#%%
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

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
                net.append(nn.Linear(C, l, bias=True))
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
                for p in m.parameters():
                    p.requires_grad = True
            else:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x, task_no):
        out = self.feature(x)
        out = self.cls[task_no](out)
        return out

#%%
def class_based_subtask(subtask_conf, ds):
    subtask_idx = [[] for _ in range(len(subtask_conf))]

    for (idx, (_, y)) in enumerate(ds):
        for (i, clses) in enumerate(subtask_conf):
            if y in clses:
                subtask_idx[i].append(idx)

    return [Subset(ds, subtask_idx[i]) for i, _ in enumerate(subtask_idx)]

def get_data():
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    ds_train = datasets.CIFAR10(root='/tmp/data', train=True, transform=T)
    ds_test = datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=T)
    subtask_conf = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    ds_train = class_based_subtask(subtask_conf, ds_train)
    ds_test = class_based_subtask(subtask_conf, ds_test)
    return ds_train, ds_test

ds_train, ds_test = get_data()
batch_size = 128
num_cls = 2

#%%
def train_task(model, task_no, device):
    from torch import optim
    train_loader = DataLoader(ds_train[task_no], batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    model.add_cls_head(2)
    model = model.to(device)
    model.prepare_for_task(task_no)

    for epoch in range(10):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, task_no)
            y -= task_no * num_cls # label offset
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test_task(model, task_no, device):
    test_loader = DataLoader(ds_test[task_no], 256, shuffle=False)
    acc, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, task_no)
            y -= task_no * num_cls # label offset
            acc += torch.sum(torch.argmax(logits, dim=1) == y).item()
            total += len(y)

    print(f"Test acc task {task_no}: {acc/total:.2f}")

model = CIL()
for task_no in range(5):
    print(f"Training task: {task_no}")
    train_task(model, task_no, 'cuda')
    for i in range(task_no + 1):
        test_task(model, i, 'cuda')

