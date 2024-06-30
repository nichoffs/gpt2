# --------- IMPORTS ---------

import math
import os
from dataclasses import dataclass
from sys import exit
from time import perf_counter_ns

import numpy as np
import tiktoken
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import fetch
from tinygrad.nn import Embedding, LayerNorm, Linear
from tinygrad.nn.optim import AdamW, dedup, OptimizerGroup
from tinygrad.nn.state import (get_parameters, get_state_dict, load_state_dict,
                               torch_load)
from tqdm import tqdm, trange

# --------- UTILS ---------

def topk(input_, k, device, dim=-1, largest=True, sorted=False):
    k = min(k, input_.shape[dim] - 1)
    input_ = input_.numpy()
    if largest:
        input_ *= -1
    ind = np.argpartition(input_, k, axis=dim)
    if largest:
        input_ *= -1
    ind = np.take(ind, np.arange(k), axis=dim)  # k non-sorted indices
    input_ = np.take_along_axis(input_, ind, axis=dim)  # k non-sorted values
    if not sorted:
        return Tensor(input_), Tensor(ind)
    if largest:
        input_ *= -1
    ind_part = np.argsort(input_, axis=dim)
    ind = np.take_along_axis(ind, ind_part, axis=dim)
    if largest:
        input_ *= -1
    val = np.take_along_axis(input_, ind_part, axis=dim)
    return Tensor(val), Tensor(ind)

def clip_grad_norm(optimizer):
    global_norm = Tensor([0.0], dtype=dtypes.float32).realize()
    for i, p in enumerate(optimizer.params):
        global_norm += p.grad.float().square().sum()
    global_norm = global_norm.sqrt()
    for i, p in enumerate(optimizer.params): p.grad = (p.grad / Tensor.where(global_norm > 1.0, global_norm, 1.0)).cast(p.grad.dtype)
    return global_norm

# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def configure_optimizers(parameters, lr, b1, b2, eps, wd):
    # TODO: do I need to include requires_grad for the count to be correct?
    # Think about this when adding bias for attention

    params_nodecay = [
        param
        for param in parameters
        if len(param.shape) < 2 and (param.requires_grad or param.requires_grad is None)
    ]
    params_decay = [
        param
        for param in parameters
        if len(param.shape) >= 2
        and (param.requires_grad or param.requires_grad is None)
    ]

    opt_decay = AdamW(params_decay, lr=lr, b1=b1, b2=b2, eps=eps, weight_decay=wd)
    opt_nodecay = AdamW(params_nodecay, lr=lr, b1=b1, b2=b2, eps=eps, weight_decay=0)

    num_params_decay = sum(param.numel() for param in opt_decay.params)
    num_params_nodecay = sum(param.numel() for param in opt_nodecay.params)

    print(
        f"num decay params: {num_params_decay} num nodecay params: {num_params_nodecay}"
    )

    optim_group = OptimizerGroup(opt_decay, opt_nodecay)

    return optim_group



# --------- CONFIG ---------


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304
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
    "gpt2": GPT2Small,
    "gpt2-medium": GPT2Medium,
    "gpt2-large": GPT2Large,
    "gpt2-xl": GPT2XL,
}

# --------- MODEL DEFINITIONS ---------


class MLP:
    def __init__(self, config: GPT2Config):
        self.c_fc = Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.RESIDUAL_SCALING = 1

    @property
    def parameters(self):
        return [self.c_fc, self.c_proj]

    def __call__(self, x):
        x = self.c_fc(x).gelu()
        x = self.c_proj(x)
        return x


class Attention:
    def __init__(self, config: GPT2Config):
        self.config = config
        self.c_attn = Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALING = 1

    @property
    def parameters(self):
        return [self.c_attn, self.c_proj]

    def __call__(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(C, dim=-1)  # (B,T,3C) -> (B,T,C) x 3
        split_heads = lambda x: x.view(
            B, T, self.config.n_head, self.config.n_embd // self.config.n_head
        ).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))

        y = q.scaled_dot_product_attention(k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class TransformerBlock:
    def __init__(self, config: GPT2Config):
        self.ln_1 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.attn = Attention(config)
        self.mlp = MLP(config)

    @property
    def parameters(self):
        return [self.ln_1, self.ln_2, *self.attn.parameters, *self.mlp.parameters]

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2:
    def __init__(self, config: GPT2Config = GPT2Small):
        self.config = config

        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, config.norm_eps)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        # init weights
        for param in self.parameters:
            self.init_weights(param)

        self.lm_head.weight = self.wte.weight

    @property
    def parameters(self):
        parameters = [self.wte, self.wpe, self.ln_f, self.lm_head]
        for block in self.h:
            parameters.extend(block.parameters)
        return parameters

    def __call__(self, idx, targets=None):
        B, T = idx.shape

        assert (
            T <= self.config.block_size
        ), f"Cannot forward, model block size is {self.config.block_size} but got sequence of length {T}"
        pos = Tensor.arange(0, T, dtype=dtypes.long)  # (T,)
        pos_emb = self.wpe(pos)  # (T,) -> (T,C)
        tok_emb = self.wte(idx)  # (B,T) -> (B,T,C)

        x = tok_emb + pos_emb
        x = x.sequential(self.h)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,C) -> (B,T,V)

        if targets is not None:
            loss = logits.flatten(0, 1).sparse_categorical_crossentropy(
                targets.flatten()
            )
            return logits, loss

        return logits, None

    @staticmethod
    def build(MODEL_NAME):
        weights = torch_load(
            fetch(f"https://huggingface.co/{MODEL_NAME}/resolve/main/pytorch_model.bin")
        )

        transposed = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )
        for k in weights:
            if k.endswith(transposed):
                weights[k] = weights[k].T

        weights["lm_head.weight"] = weights["wte.weight"]
        model = GPT2(MODEL_CONFIGS[MODEL_NAME])
        load_state_dict(model, weights)

        return model

    def init_weights(self, param):
        if isinstance(param, Linear):
            std = 0.02
            if hasattr(param, "RESIDUAL_SCALING"):
                std *= (2 * self.config.n_layer) ** -0.5
            param.weight = Tensor.normal(
                param.weight.shape,
                mean=0,
                std=std,
            )
            if param.bias is not None:
                param.bias = Tensor.zeros_like(param.bias)
        elif isinstance(param, Embedding):
            param.weight = Tensor.normal(param.weight.shape, mean=0, std=0.02)


# --------- DATALOADER ---------


class DataLoaderLite:
    def __init__(self, B, T, file_path):
        self.B = B
        self.T = T

        self.batch = lambda x: x.reshape(B, T)

        with open(file_path, "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")

        tokens = enc.encode(text)
        self.tokens = np.array(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = self.batch(buf[:-1])
        y = self.batch(buf[1:])
        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            print("read entire document, resetting position...")
            self.current_position = 0

        return x, y


# --------- INITIALIZATION ---------

model = GPT2(GPT2Small)
optim = configure_optimizers(get_parameters(model), 3e-4, .9, .95, 1e-8, .1)
dl = DataLoaderLite(4, 128, "datasets/shake.txt")

# --------- TRAINING ---------

@TinyJit
def train_step(x, y):
    with Tensor.train():
        optim.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        # norm = clip_grad_norm(optim)
        optim.step()
        return loss.realize(), 0


for step in (t := trange(max_steps)):
    t0 = perf_counter_ns()
    x, y = dl.next_batch()
    lr = get_lr(step)
    for opt in optim.optimizers:
        opt.lr = lr
    loss, norm = train_step(Tensor(x, dtype=dtypes.long), Tensor(y, dtype=dtypes.long))
    loss = loss.item()
    dt = (perf_counter_ns() - t0) * 1e-6
    t.set_description(
        f"train loss: {loss:.2f} | dt: {dt:.2f} | tok/s {(dl.B*dl.T)/(dt*1e-3):.2f} | lr: {lr:.5f} | norm: {norm:.2f}"
    )
