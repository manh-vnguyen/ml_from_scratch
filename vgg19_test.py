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
                net.append(nn.Conv2d(C, l, kernel_size=3, padding=1, bias=True))
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
                net.append(nn.Linear(C, l, bias=True))
                C = l
            if isinstance(l, float):
                net.append(nn.Dropout(l))
            elif l == 'norm':
                net.append(nn.BatchNorm1d(C))
            elif l == 'act':
                net.append(nn.ReLU(inplace=True))
                

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
    
def vgg19(num_classes, C, H, W):
    conv = [64, 'norm', 'act', 64, 'norm', 'act', 'pool',
            128, 'norm', 'act', 128, 'norm', 'act', 'pool',
            256, 'norm', 'act', 256, 'norm', 'act', 256, 'norm', 'act', 256, 'norm', 'act', 'pool',
            512, 'norm', 'act', 512, 'norm', 'act', 512, 'norm', 'act', 512, 'norm', 'act', 'pool',
            512, 'norm', 'act', 512, 'norm', 'act', 512, 'norm', 'act', 512, 'norm', 'act', 'pool',]
    cls = [4096, 0.5, 'act', 4096, 0.5, 'act', num_classes]
    return VGG(conv, cls, C, H, W)

def count_params(model):
    count = 0
    for n, p in model.named_parameters():
        # print(n, p.shape)
        count += p.numel()
    return count

from tqdm.auto import tqdm

def eval_model(model, loader, device, criterion):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    pbar = tqdm(loader, desc=f"Progress:", dynamic_ncols=True)
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total

from dotenv import load_dotenv
import os
load_dotenv('.env')

if __name__ == '__main__':

    root = os.environ.get('IMAGENET_DIR')
    transform = transforms.Compose([
        transforms.Resize(256),                            # keep aspect ratio; shorter side -> 256
        transforms.CenterCrop(224),                        # fixed 224x224 crop
        transforms.ToTensor(),                             # PIL [0..255] -> float tensor [0..1], CxHxW
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet channel-wise normalization
                            std=[0.229, 0.224, 0.225]),
    ])

    test_ds = datasets.ImageNet(root=root, split="val", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=4, persistent_workers=True, pin_memory=True)

    from torchvision import models

    # model = models.vgg19_bn(weights=models.VGG19_BN_Weights).to('cuda')

    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2).to('cuda')

    test_loss, test_acc = eval_model(model, test_loader, 'cuda', nn.CrossEntropyLoss())

    print(f"{test_loss:.2f} {test_acc * 100:.2f}")
