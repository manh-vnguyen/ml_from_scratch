#%%
import torch
from utils import BatchNorm

class Client():
    def __init__(self, model_type, model, trainloader, loss_fn, device, checkpointed_epoch=0):
        assert model_type in ('snn', 'ann')
        self.model_type = model_type
        self.device = device
        self.model = model
        self.trainloader = trainloader
        self.loss_fn = loss_fn
        self.data_iter = iter(self.trainloader)
        for _ in range(checkpointed_epoch % len(self.trainloader)):
            next(self.data_iter)

    def train(self):
        try:
            x, y = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.trainloader)
            x, y = next(self.data_iter)
        x, y = x.to(self.device), y.to(self.device)
        self.model.zero_grad()
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.loss = loss.item()
        self.num_samples = y.shape[0]

def test_model(model, loader, device):
    acc, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc += torch.sum(torch.argmax(logits, dim=1) == y).item()
            total += y.shape[0]
    test_acc = acc/total*100
    return test_acc

def eval_model(model, loader, device, criterion=torch.nn.CrossEntropyLoss()):
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