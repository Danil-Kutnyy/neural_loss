import torch
import numpy as np
import sys
import random
import os
import logging
import matplotlib.pyplot as plt
import pickle
import time
from torch import nn
from torch.optim import Adam, AdamW
import torch.nn.functional as F
import copy
import glob
import gc
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import math


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 10
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
#print('device:',device)
eval_iters = 200
n_embd = 32
n_head = 3
n_layer = 3
dropout = 0.15
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/dataset/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        predictions: Tensor of shape (B, T, C)
        targets: Tensor of shape (B, T), where values are class indices (not one-hot encoded)
        """
        # Reshape predictions to be 2D: (B * T, C)
        B, T, C = predictions.shape
        predictions = predictions.view(B * T, C)

        # Reshape targets to be 1D: (B * T)
        targets = targets.view(-1)

        # Apply softmax to the predictions to get probabilities
        softmax_preds = F.softmax(predictions, dim=1)
        
        # Create one-hot encoding for the target labels
        target_one_hot = F.one_hot(targets, C)

        # Compute cross-entropy loss manually
        loss = -torch.sum(target_one_hot * torch.log(softmax_preds + 1e-10), dim=1)

        # Return mean loss
        return loss
class LNN(torch.nn.Module):
    def __init__(self, block, classes):
        super().__init__()
        self.classes = classes
        self.block = block
        self.gelu = nn.GELU()
       
        #locally conected
        out_size = 4
        input_size = 2
        self.lc1 = nn.Linear(input_size, 8)
        self.lc21 = nn.Linear(8, out_size)
        self.lc22 = nn.Linear(8, out_size)
        
        size = classes*block

        #fully token conected
        #token-wise comunication
        self.ftc1 = nn.Linear(out_size*classes, classes)  # First fully connected layer
        #token compression for further block-wise cominciation
        self.ftc2 = nn.Linear(out_size*classes, 2) # Second fully connected layer

        #fully block connected
        self.fbc1 = nn.Linear(block*2, 4)

        self.lc3 = nn.Linear(4+1+input_size, 32)
        self.lc4 = nn.Linear(32, 16)
        self.lc5 = nn.Linear(16, 1)

    def forward(self, inp):
      x = self.lc1(inp)
      pre = self.gelu(x)
      x1 = self.lc21(pre)
      x2 = self.lc22(pre)
      x = self.gelu(x1)

      B, T, C, F  = x.shape
      ftc_inp = x.view(B, T, C*F)

      ftc_out = self.gelu( self.ftc1(ftc_inp) )
      ftc_out = ftc_out.view(B, T, C, 1)

      fbc_inp =  self.gelu( self.ftc2(x2.view(B, T, C*F)) )
      fbc_inp = fbc_inp.view(B, T*2)
      
      fbc_out = self.gelu( self.fbc1(fbc_inp))
      fbc_out = fbc_out.reshape(B, 1, 1, 4).expand(B, T, C, 4)
      pre_data = torch.concat((fbc_out, ftc_out, inp), dim=-1)

      
      x = self.gelu(self.lc3(pre_data))
      x = self.gelu(self.lc4(x))
      y = torch.nn.ReLU()(self.lc5(x))
      return y

class NNLoss(torch.nn.Module):
    def __init__(self, block, classes, load_weigths=True, add_noise=False):
        super().__init__()
        self.lnn = LNN(block, classes)
        self.load_cross_entropy_weigths(load_weigths, add_noise)

    def forward(self, x, y):
      x = F.softmax(x, dim=-1)
      y = torch.nn.functional.one_hot(y, self.lnn.classes)
      cat_input = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)
      raw_loss = self.lnn(cat_input).squeeze()
      loss = raw_loss.sum(-1)
      loss = loss.view(-1)
      #return loss
      return loss.mean()

    def load_cross_entropy_weigths(self, load_weigths=True, add_noise=False):
        #self.to('cpu')
        if load_weigths:
            self.load_state_dict(torch.load('/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/loss_nn/model3_mps', weights_only=True))
        
        #self.to(device)
        if add_noise:
            for p in self.parameters():
                p.data = p.data + torch.randn(p.data.shape)*(random.randint(1,20)/1000)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, load_loss_weigths=True, add_loss_noise=False):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)
        self.loss = NNLoss(block_size, vocab_size, load_weigths=load_loss_weigths, add_noise=add_loss_noise)
        self.loss_weigths_freeze()
        #self.loss.load_state_dict(torch.load('/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/loss_nn/model2_mps', weights_only=True))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def loss_weigths_freeze(self):
        for param in self.loss.parameters():
            param.requires_grad = False

    def set_loss_weights(self, new_weigths):
        #print(new_weigths.keys())
        #print(self.state_dict().keys())
        #print('new w:')
        #for n, w in new_weigths.items():
        #    print(w)
        #    break
        ##print('before:')
        #for n, w in self.state_dict().items():
        #    if n[5:] in new_weigths.keys():
        #        #print(w)
        #        break
        old_state_dict = self.state_dict()
        new_weigths_all = {}
        for n, w in old_state_dict.items():
            if n[5:] in new_weigths.keys():
                #print('here!')
                new_weigths_all[n] = new_weigths[n[5:]].detach().clone()
            else:
                new_weigths_all[n] = w.detach().clone()
                #print('changed!')
        self.load_state_dict(new_weigths_all)
        
        #print('after')
        #for n, w in self.state_dict().items():
        #    if n[5:] in new_weigths.keys():
        #        print(w)
        #        break
        #sys.exit()
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            #data = [logits.clone().detach(), targets.clone().detach()]
            #old custom loss
            logits2 = logits.view(B*T, C)
            targets2 = targets.view(B*T)
            loss2 = F.cross_entropy(logits2, targets2)

            #new nn loss
            logits = logits
            loss = self.loss(logits, targets)

        return logits, loss, loss2

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

'''
model = GPTLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()), 'parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#logits_list = saved_data['logits']
#targets_list = saved_data['targets']
#saved_data = torch.load('/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/loss_nn_training_set/logits_targets_data.pt')

logits_list = []
targets_list = []
losses_val_data = []
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        losses_val_data.append(losses['val'])
        #torch.save({'logits': data[0], 'targets': data[1]}, f'/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/loss_nn_training_set/logits_targets_data{iter+2}.pt')

    # sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss, loss2 = model(xb, yb)
    print(loss.item(),loss2.item())
    
    #logits_list.append(data[0])
    #targets_list.append(data[1])
    
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


plt.plot(losses_val_data, label='val_loss')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('val loss result')

# Add legend
plt.legend()

# Show the plot
plt.show()
'''





























