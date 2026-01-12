# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import einops
from einops.layers.torch import Rearrange

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_ds = datasets.CIFAR10(root="/tmp/data", train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10(root="/tmp/data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

torch.manual_seed(37)

class LayerNorm(nn.Module):
    VARIANCE_UNBIASED = True # False will make it identical to nn.LayerNorm
    def __init__(self, norm_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm_ndim = 1 if type(norm_shape) is int else len(norm_shape)
        self.gamma = nn.Parameter(torch.ones(norm_shape))
        self.beta = nn.Parameter(torch.zeros(norm_shape)) 

    def forward(self, x):
        mean = torch.mean(x, dim=list(range(-self.norm_ndim, 0)), keepdim=True) # (*, *norm_shape) 
        var = torch.var(x, dim=list(range(-self.norm_ndim, 0)), keepdim=True, unbiased=self.VARIANCE_UNBIASED) # (*, *norm_shape)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.gamma * x - self.beta # x -> (*, *norm_shape); self.gamma -> (*norm_shape) will be broadcasted
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, C, P, D):
        super().__init__()
        self.n_embd = P
        self.proj = nn.Sequential(
            # LayerNorm(C*P*P),
            nn.Linear(in_features=C*P*P, out_features=D),
            # LayerNorm(D),
        )

    def forward(self, x): # B, C, H, W
        x = einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
                             p1=self.n_embd, p2=self.n_embd) # B, N, C*P*P
        return self.proj(x)

class PatchEmbeddingUniqueProjection(nn.Module):
    def __init__(self, N, C, P, D):
        super().__init__()
        self.patch_size = P
        self.proj_W = nn.Parameter(torch.randn(N, D, C*P*P))
        self.proj_b = nn.Parameter(torch.randn(N, D))

    def forward(self, x): # B, C, H, W
        x = einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
                             p1=self.patch_size, p2=self.patch_size) # B, N, C*P*P
        #x = torch.stack([x[:, i] @ self.proj_W[i].T + self.proj_b[i] for i in range(N)], dim=1)
        x = torch.einsum('bnd,nod->bno', x, self.proj_W) + self.proj_b.unsqueeze(0)  # B, N, D
        return x

#%%

class FeedForward(nn.Module):
    def __init__(self, D, n_hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(D),
            nn.Linear(D, n_hidden),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(n_hidden, D),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
class Encoder(nn.Module):
    def __init__(self, D, n_heads, n_hidden, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=n_heads, batch_first=True, dropout=dropout, bias=False)
        self.ln1 = self.ln = LayerNorm(D)
        self.ffwd = FeedForward(D, n_hidden, dropout)
        self.ln2 = self.ln = LayerNorm(D)

    def forward(self, x):
        ln1_x = self.ln1(x)
        x = x + self.attn(ln1_x,ln1_x,ln1_x)[0]
        x = x + self.ffwd(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self, H, W, C, P, D, n_heads, n_hidden, L, n_classes, dropout):
        super().__init__()
        assert H % P == 0 and W % P == 0, "P must be a factor of H and W"
        N = H * W // P**2 # Add the classification patch
        self.n_embd = D
        # self.register_buffer('pos_enc', ViT.positional_encoding(N, D))
        self.pos_embd = nn.Parameter(torch.randn(N + 1, D))
        self.patch_embd = PatchEmbedding(C, P, D)
        self.encoders = nn.Sequential(*[Encoder(D, n_heads, n_hidden, dropout) for _ in range(L)])
        self.cls_patch_embd = nn.Parameter(torch.randn(D))
        self.dropout = nn.Dropout(dropout)
        self.cls_ln = LayerNorm(D)
        self.classifier = nn.Linear(D, n_classes)

    def forward(self, x): # B, C, H, W
        img_patches = self.patch_embd(x)
        out = torch.cat([ self.cls_patch_embd.broadcast_to(x.size(0), 1, self.n_embd), img_patches ], dim=1) # (B, seq_len = N + 1, D)
        out = out + self.pos_embd
        out = self.dropout(out)
        out = self.encoders(out)
        out = self.classifier(self.cls_ln(out[:, 0, :].squeeze(1)))
        return out

#%%

def get_parameter_groups(model, weight_decay):
    """
    Separate parameters into two groups: those that should have weight decay
    and those that shouldn't (LayerNorm parameters: gamma and beta).
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if this is a LayerNorm parameter (gamma or beta)
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

def train(model, epochs, device, wandb_run):
    # Model, loss, optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # param_groups = get_parameter_groups(model, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, eta_min=1e-6)

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
    for p in model.parameters():
        count += p.numel()
    return count

# %%
if __name__ == '__main__':
    import wandb
    
    device = 'cuda'
    model = ViT(H=32, W=32, C=3, P=1, D=64, n_heads=8, n_hidden=256, L=6, n_classes=10, dropout=0.1).to(device)
    run = wandb.init(project='ViT', 
                     notes="ViT(H=32, W=32, C=3, P=2, D=64, n_heads=8, n_hidden=256, L=6, n_classes=10, dropout=0.1)"
                    )
    print(model)
    run.tags += (f"count_params={count_params(model)}",)
    train(model, 100,'cuda', wandb_run=run)
    run.finish()
