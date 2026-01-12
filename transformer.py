import torch
from torch import nn
from torch.nn import functional as F

with open('input.txt', 'r') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
for ch in chars:
    assert itos[stoi[ch]] == ch
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
assert decode(encode(text[100-130])) == text[100-130]

data = torch.tensor(encode(text))

# Split train and test data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(1337)
batch_size = 128 # how many independent sequences will be processed in parallel?
block_size = 32 # what is the maximum context length for prediction
import random
def get_example(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    i = random.randint(0, len(data) - block_size)
    x = data[i: i + block_size]
    y = data[i + 1: i + block_size + 1]
    return x, y

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y


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

torch.set_default_device('cpu')

import math

class SelfAttention(nn.Module):
    mask = None
    @staticmethod
    def attention(q, k, v): # q, k, v -> (B, seq_len, head_size)
        head_size = q.shape[-1]
        attn_weight = (q @ k.transpose(-2, -1))/math.sqrt(head_size)
        if SelfAttention.mask != None:
            attn_weight.masked_fill_(SelfAttention.mask == 0, float('-inf'))
        attn_weight = F.softmax(attn_weight, dim=-1) # (B, seq_len, seq_len)
        return attn_weight @ v # (B, seq_len, seq_len) @ (B, seq_len, head_size) -> (B, seq_len, head_size)
    
    def __init__(self, d_model, head_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x): # (B, seq_len, d_model)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return SelfAttention.attention(q, k, v) # (B, seq_len, head_size)

    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "Number of attn heads must be divisible by d_model"
        head_size = d_model // num_heads
        self.num_heads = num_heads
        SelfAttention.mask = torch.ones(block_size, block_size, dtype=torch.bool).tril()
        self.attn_heads = nn.ModuleList([SelfAttention(d_model, head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.attn_heads], dim=-1)
        return self.projection(out)

def residual(feature, ln, x):
    # Original implementation: Add residual then norm
    return ln(x + feature(x))

class FeedForwardBlock(nn.Module):
    BLOAT = 4
    def __init__(self, d_model):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(d_model, self.BLOAT * d_model),
            nn.ReLU(),
            nn.Linear(self.BLOAT * d_model, d_model),
        )
        self.ln = LayerNorm(d_model)

    def forward(self, x):
        return residual(self.ffwd, self.ln, x)

class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.multi_head = MultiHeadAttention(d_model, num_head)
        self.ln = LayerNorm(d_model)

    def forward(self, x):
        return residual(self.multi_head, self.ln, x)
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.attn_block = AttentionBlock(d_model, num_head)
        self.ffwd_block = FeedForwardBlock(d_model)

    def forward(self, x):
        return self.ffwd_block(self.attn_block(x))

class LM(nn.Module):
    @staticmethod
    def positional_encoding(seq_len=32, d_model=64, freq_constant=25):
        pos = torch.arange(1, seq_len + 1)
        pos_enc = torch.empty(seq_len, d_model)
        wave_freq_scaler = freq_constant ** (torch.arange(0, d_model, 2) / d_model)
        pos_enc[:, ::2] = torch.sin( pos.unsqueeze(1) / wave_freq_scaler[:(d_model - 1) // 2 + 1].unsqueeze(0) )
        pos_enc[:, 1::2] = torch.cos( pos.unsqueeze(1) / wave_freq_scaler[:d_model // 2].unsqueeze(0) )
        return pos_enc
    
    def __init__(self, seq_len, vocab_size, d_model, num_head, num_layers, positional_encoding_freq_constant):
        super().__init__()
        self.seq_len = seq_len
        self.w_embd = nn.Embedding(vocab_size, d_model)
        if positional_encoding_freq_constant != None:
            self.pos_enc = LM.positional_encoding(seq_len, d_model, positional_encoding_freq_constant)
            self.positional_encoding = True
        else:
            self.pos_embd = nn.Embedding(seq_len, d_model)
            self.positional_encoding = False
        
        self.encoder_blocks = nn.Sequential(
            *[Encoder(d_model, num_head) for _ in range(num_layers)]
        )
        self.pred = nn.Linear(d_model, vocab_size)

    def forward(self, inputs): # B, seq_len
        assert inputs.shape[1] == block_size, f"seq_len is not {self.seq_len}"
        x = self.w_embd(inputs) # B, seq_len, d_model
        if self.positional_encoding:
            x = x + self.pos_enc
        else:
            x = x + self.pos_embd(torch.arange(self.seq_len))      
        x = self.encoder_blocks(x)
        return self.pred(x)
    
    def generate(self, prefix, num_tokens):
        if len(prefix) < self.seq_len:
            inputs = encode(' ' * (self.seq_len - len(prefix)) + prefix)
        else:
            inputs = encode(prefix[-self.seq_len:])
        inputs = torch.tensor([inputs]).to(torch.get_default_device())
        out_tokens = []
        with torch.no_grad():
            for _ in range(num_tokens):
                outputs = self(inputs)
                probs = F.softmax(outputs.squeeze(0)[-1], 0)
                idx_next = torch.multinomial(probs, 1).item()
                out_tokens.append(idx_next)
                inputs = torch.cat([inputs[:, 1:], torch.tensor(idx_next).view(inputs.size(0), 1)], dim=1) # (B, seq_len - 1) + (B, 1)
        return prefix + decode(out_tokens)

    


def count_params(model):
    count = 0
    for p in model.parameters():
        count += p.numel()
    print("Number of params:", count)


eval_iters = 200
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            Y_hat = model(X)
            losses[k] = F.cross_entropy(Y_hat.view(-1, vocab_size), Y.view(-1))
        out[split] = losses.mean().item()
    return out

def train(lm, n_batches, lr, model_name='transformer.pth'):
    optimizer = torch.optim.AdamW(lm.parameters(), lr=lr)
    for i in range(n_batches):
        xb, yb = get_batch('train')
        xb, yb = xb.to(torch.get_default_device()), yb.to(torch.get_default_device())
        outputs = lm(xb)
        loss = F.cross_entropy(outputs.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        # print(max([p.grad.abs().max().item() if p.data != None else None for p in lm.parameters()]))
        optimizer.step()
        if (i + 1) % 1000 == 0:
            print(i, loss.item())
            print(lm.generate(' ', 2000))
            val = estimate_loss(lm)
            print(val)
            wandb.log(val)
            if model_name != None:
                torch.save({'epoch': i, 'state': lm.state_dict()}, model_name)

import wandb
if __name__ == "__main__":
    lm = LM(seq_len=block_size, vocab_size=vocab_size, d_model=64, num_head=8, num_layers=2, positional_encoding_freq_constant=12)
    wandb.init(project='transformer_positional_encoding')
    print(lm)
    count_params(lm)
    train(lm, 20000, 1e-2)
    train(lm, 20000, 1e-3)
    train(lm, 20000, 1e-4)
