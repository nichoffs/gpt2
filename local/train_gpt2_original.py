from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.helpers import GlobalCounters, Timing
import contextlib
from time import perf_counter
from tinygrad.nn import Embedding, Linear, LayerNorm
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict, get_parameters
from tqdm import tqdm, trange
from tinygrad.nn.optim import AdamW
from dataclasses import dataclass
from tinygrad.helpers import fetch
import tiktoken
import time
import numpy as np
import os
import matplotlib.pyplot as plt




@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    norm_eps: float = 1e-5

@dataclass
class GPT2Small(GPT2Config):
    pass

@dataclass
class GPT2Medium(GPT2Config):
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024

@dataclass
class GPT2Large(GPT2Config):
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 1280

@dataclass
class GPT2XL(GPT2Config):
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600

MODEL_CONFIGS = {
    'gpt2': GPT2Small,
    'gpt2-medium': GPT2Medium,
    'gpt2-large': GPT2Large,
    'gpt2-xl': GPT2XL
}

class MLP:
    def __init__(self, config : GPT2Config):
        self.c_fc = Linear(config.n_embd, config.n_embd*4)
        self.c_proj = Linear(config.n_embd*4, config.n_embd)
    def __call__(self, x):
        x = self.c_fc(x).gelu()
        x = self.c_proj(x)
        return x

class Attention:
    def __init__(self, config: GPT2Config):
        self.config = config
        self.c_attn = Linear(config.n_embd, config.n_embd*3)
        self.c_proj = Linear(config.n_embd, config.n_embd)
    def __call__(self, x):
        B,T,C = x.shape

        q, k, v = self.c_attn(x).split(C, dim=-1) #(B,T,3C) -> (B,T,C) x 3
        split_heads = lambda x: x.view(B, T, self.config.n_head, self.config.n_embd//self.config.n_head).transpose(1,2)
        q, k, v = map(split_heads, (q,k,v))

        y = q.scaled_dot_product_attention(k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

class TransformerBlock:
    def __init__(self, config : GPT2Config):
        self.ln_1 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.attn = Attention(config)
        self.mlp = MLP(config)
    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2:
    def __init__(self, config : GPT2Config = GPT2Small):
        self.config = config

        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, config.norm_eps)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        # tie weights - HUGE SAVINGS
        self.lm_head.weight = self.wte.weight

    def __call__(self, idx, targets=None):
        B,T = idx.shape

        assert T <= self.config.block_size, f"Cannot forward, model block size is {self.config.block_size} but got sequence of length {T}"
        pos = Tensor.arange(0, T, dtype=dtypes.long) # (T,)
        pos_emb = self.wpe(pos) # (T,) -> (T,C)
        tok_emb = self.wte(idx) # (B,T) -> (B,T,C)

        x = tok_emb + pos_emb
        x = x.sequential(self.h)

        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,C) -> (B,T,V)

        if targets is not None:
            loss = logits.flatten(0,1).sparse_categorical_crossentropy(targets.flatten())
            return logits, loss

        return logits, None

    @staticmethod
    def build(MODEL_NAME):

        weights = torch_load(fetch(f'https://huggingface.co/{MODEL_NAME}/resolve/main/pytorch_model.bin'))

        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
        for k in weights:
            if k.endswith(transposed):
                weights[k] = weights[k].T

        weights['lm_head.weight'] = weights['wte.weight']
        model = GPT2(MODEL_CONFIGS[MODEL_NAME])
        load_state_dict(model, weights)

        return model

class DataLoaderLite:
    def __init__(self, B, T, file_path):
        self.B=B
        self.T=T

        self.batch = lambda x: x.reshape(B,T)

        with open(file_path, 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')

        tokens = enc.encode(text)
        self.tokens = np.array(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position + B*T+1]
        x = self.batch(buf[:-1])
        y = self.batch(buf[1:])
        self.current_position += B*T

        if self.current_position + (B*T+1) > len(self.tokens):
            print("read entire document, resetting position...")
            self.current_position = 0

        return x,y


model = GPT2(GPT2Small)
optim = AdamW(get_parameters(model), lr=3e-4)
dl = DataLoaderLite(2, 1024, "datasets/shake.txt")

@TinyJit
def train_step(x,y):
    with Tensor.train():
        optim.zero_grad()
        logits, loss = model(x,y)
        loss.backward()
        optim.step()
        return loss


for i in (t := trange(1000)):
    t0 = time.perf_counter_ns()
    x, y = dl.next_batch()
    loss = train_step(Tensor(x,dtype=dtypes.long),Tensor(y,dtype=dtypes.long)).item()
    et = (time.perf_counter_ns() - t0)*1e-6
    t.set_description(f"train loss: {loss:.2f} | dt: {et} | tok/s {(dl.B*dl.T)/(et*1e-3)}")
