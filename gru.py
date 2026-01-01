# %%
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device('cuda')

with open('input.txt', 'r') as f:
    text = f.read()


print("Length of the dataset: ", len(text))


# for char in "\n!$&',-.3:;?":
#     if char == '\n':
#         text = text.replace(char, ' ')
#     else:
#         text = text.replace(char, '')

print(text[:1000])


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

# %%
data = torch.tensor(encode(text))

# Split train and test data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 128 # how many independent sequences will be processed in parallel?
block_size = 32 # what is the maximum context length for prediction

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y

eval_iters = 200
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.cuda(), Y.cuda()
            Y_hat, _ = model(X, None)
            losses[k] = F.cross_entropy(Y_hat.view(-1, vocab_size), Y.view(-1))
        out[split] = losses.mean().item()
    return out

# %%
class RNNLMScratch(nn.Module):
    def __init__(self, rnn, hidden_size, vocab_size, sigma):
        super().__init__()
        self.rnn = rnn
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W_hq = nn.Linear(hidden_size, vocab_size)

    def clip_gradient(self):
        params = [p for p in self.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
        if norm > 1:
            for p in params:
                p.grad[:] /= norm

    def forward(self, inputs, state):
        inputs = F.one_hot(inputs, self.vocab_size).float().to(device)
        outputs, state = self.rnn(inputs, state)
        outputs = self.W_hq(outputs)
        return outputs, state
    
    def generate(self, prefix, state, num_tokens):
        inputs = encode(prefix)
        out_tokens = []
        with torch.no_grad():
            for _ in range(num_tokens):
                outputs, state = self(torch.tensor([inputs]), state)
                probs = F.softmax(outputs[-1][0], 0)
                idx_next = torch.multinomial(probs, 1).item()
                out_tokens.append(idx_next)
                inputs = [idx_next]
        return prefix + decode(out_tokens)

# %%
class GRUScratch(nn.Module):
    def __init__(self, input_size, hidden_size, sigma=0.01):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Reset gate
        self.W_xr = nn.Parameter(torch.randn(input_size, hidden_size) * sigma)
        self.W_hr = nn.Parameter(torch.randn(hidden_size, hidden_size) * sigma)
        self.b_r = nn.Parameter(torch.randn(hidden_size))
        # Update gate
        self.W_xz = nn.Parameter(torch.randn(input_size, hidden_size) * sigma)
        self.W_hz = nn.Parameter(torch.randn(hidden_size, hidden_size) * sigma)
        self.b_z = nn.Parameter(torch.randn(hidden_size))
        # Candidate for the new hidden state
        self.W_xc = nn.Parameter(torch.randn(input_size, hidden_size) * sigma)
        self.W_hc = nn.Parameter(torch.randn(hidden_size, hidden_size) * sigma)
        self.b_c = nn.Parameter(torch.randn(hidden_size))

    def forward(self, inputs, H):
        # inputs -> (B, N, input_size), H_C -> (B, N, hidden_size), (B, N, hidden_size)
        if H is None:
            H = torch.zeros(inputs.shape[0], self.hidden_size).to(inputs.device) # (B, hidden_size)

        inputs = inputs.transpose(0, 1) # (N, B, hidden_size)
        outputs = []
        for x in inputs:
            r = F.sigmoid(x @ self.W_xr + H @ self.W_hr + self.b_r) # How much of old state (H_{t-1}) to incorporate into the candidate of the new hidden state (H~_t)
            z = F.sigmoid(x @ self.W_xz + H @ self.W_hz + self.b_z) # Decide the ratio of H_{t-1}:H~_t that will make up H_t
            c = F.tanh(x @ self.W_xc + r @ self.W_hc + self.b_c)    # Candidate of the new hidden state (H~_t)
            H = H * z + (1 - z) * c
            outputs.append(H)
        outputs = torch.stack(outputs).transpose(0, 1)
        return outputs, H

# %%
gru = GRUScratch(input_size=vocab_size, hidden_size=128, sigma=0.01).to(device)
lm = RNNLMScratch(rnn=gru, hidden_size=128, vocab_size=vocab_size, sigma=0.01).to(device)

model_name = 'gru.pth'
# lm.load_state_dict(torch.load(model_name)['state'])

import wandb

wandb.init(project='gru_tinyml')

optimizer = torch.optim.AdamW(lm.parameters(), lr=1e-3)
for i in range(10000):
    xb, yb = get_batch('train')
    xb, yb = xb.to('cuda'), yb.to('cuda')
    outputs, _ = lm(xb, None)
    loss = F.cross_entropy(outputs.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    # lm.clip_gradient()
    optimizer.step()
    if (i + 1) % 1000 == 0:
        print(i, loss.item(), estimate_loss(lm))
        print(lm.generate(' ', None, 2000))
        torch.save({'epoch': i, 'state': lm.state_dict()}, model_name)

optimizer = torch.optim.AdamW(lm.parameters(), lr=1e-4)
for i in range(10000):
    xb, yb = get_batch('train')
    xb, yb = xb.to('cuda'), yb.to('cuda')
    outputs, _ = lm(xb, None)
    loss = F.cross_entropy(outputs.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    # lm.clip_gradient()
    optimizer.step()
    if (i + 1) % 1000 == 0:
        print(i, loss.item())
        wandb.log(estimate_loss(lm))
        gen = lm.generate(' ', None, 2000)
        print(gen)
        wandb.log({"Sample": gen})
        torch.save({'epoch': i, 'state': lm.state_dict()}, model_name)
