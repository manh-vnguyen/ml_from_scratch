
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


import torch
data = torch.tensor(encode(text))


print(data.shape, data.dtype)
print(data[:1000])


# Split train and test data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


block_size = 8
train_data[:block_size + 1]


x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"When input is {context} then target: {target}")


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

import torch
from torch import nn
from torch.nn import functional as F


class RNNLMScratch(nn.Module):
    def __init__(self, rnn, num_hiddens, vocab_size, sigma):
        super().__init__()
        self.rnn = rnn
        self.num_hiddens = num_hiddens
        self.vocab_size = vocab_size
        self.W_hq = nn.Linear(num_hiddens, vocab_size)

    def clip_gradient(self):
        params = [p for p in self.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
        if norm > 1:
            for p in params:
                p.grad[:] /= norm

    def forward(self, inputs, state):
        inputs = F.one_hot(inputs, self.vocab_size).float().cuda()
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

class LSTMScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        # Forget gate (how much of the long-term memory is retained)
        self.W_xf = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hf = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_f = nn.Parameter(torch.randn(num_hiddens))
        # Input gate
        self.W_xi = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hi = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_i = nn.Parameter(torch.randn(num_hiddens))
        # Input node
        self.W_xc = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hc = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_c = nn.Parameter(torch.randn(num_hiddens))
        # Output gate
        self.W_xo = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_ho = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_o = nn.Parameter(torch.randn(num_hiddens))

    def forward(self, inputs, H_C):
        # inputs -> (B, N, num_inputs), H_C -> (B, N, num_hiddens), (B, N, num_hiddens)
        if H_C is None:
            H = torch.zeros(inputs.shape[0], self.num_hiddens).to(inputs.device) # (B, num_hiddens)
            C = torch.zeros(inputs.shape[0], self.num_hiddens).to(inputs.device) # (B, num_hiddens)
        else:
            H, C = H_C

        inputs = inputs.transpose(0, 1) # (N, B, num_hiddens)
        outputs = []
        for x in inputs:
            f = F.sigmoid(x @ self.W_xf + H @ self.W_hf + self.b_f) # Forget some of the long-term memory (C), 
                                                                    # which is decided by the short-term input (H, x)
            i = F.sigmoid(x @ self.W_xi + H @ self.W_hi + self.b_i) # Decide how much of short-term memory (c) 
                                                                    # to be incorporated in to long-term memory (C),
                                                                    # decided by the short-term input (H, x)
            c = F.tanh(x @ self.W_xc + H @ self.W_hc + self.b_c)    # Calculate the short-term memory to be incorporated (c) in to the long-term memory (C)
                                                                    # decided by the short-term input (H, x)
            C = C * f + i * c   # Update memory cell internal state, i.e. calculate the new long-term memory (C)
            o = F.sigmoid(x @ self.W_xo + H @ self.W_ho + self.b_o) # Calculate how much of the new long-term memory (C) will be come the new short-term memory (H)
            H = F.tanh(C) * o   # Calculate the new short-term memory, or output hidden state (H)
            outputs.append(H)
        outputs = torch.stack(outputs).transpose(0, 1)
        return outputs, (H, C)


if __name__ == '__main__':
    model_name = 'lstm_full_chars.pth'

    lstm = LSTMScratch(num_inputs=vocab_size, num_hiddens=128, sigma=0.01).cuda()
    lm = RNNLMScratch(rnn=lstm, num_hiddens=128, vocab_size=vocab_size, sigma=0.01).cuda()
    # lm.load_state_dict(torch.load(model_name)['state'])

    optimizer = torch.optim.AdamW(lm.parameters(), lr=1e-3)
    for i in range(100000):
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
            print(lm.generate(' ', None, 2000))
            torch.save({'epoch': i, 'state': lm.state_dict()}, model_name)


    optimizer = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    for i in range(100000):
        xb, yb = get_batch('train')
        xb, yb = xb.to('cuda'), yb.to('cuda')
        outputs, _ = lm(xb, None)
        loss = F.cross_entropy(outputs.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        # lm.clip_gradient()
        optimizer.step()
        if (i + 1) % 1000 == 0:
            print(loss.item())
            print(lm.generate(' ', None, 2000))
            torch.save({'epoch': i + 100000, 'state': lm.state_dict()}, model_name)