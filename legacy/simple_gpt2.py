""""
This is a simplified version of the full-scale GPT-2 training. It's meant to train on the
"""
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
from tinygrad.nn.optim import AdamW, OptimizerGroup, dedup
from tinygrad.nn.state import (
    get_parameters,
    get_state_dict,
    load_state_dict,
    torch_load,
)
from tqdm import tqdm, trange

# --------- UTILS ---------


def topk(input_, k, dim=-1, largest=True, sorted=False):
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
    for i, p in enumerate(optimizer.params):
        p.grad = (p.grad / Tensor.where(global_norm > 1.0, global_norm, 1.0)).cast(
            p.grad.dtype
        )
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
    return Tensor(min_lr + coeff * (max_lr - min_lr), requires_grad=False)


def configure_optimizers(parameters, lr, b1, b2, eps, wd):
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
optim = configure_optimizers(get_parameters(model), 3e-4, 0.9, 0.95, 1e-8, 0.1)
B = 4
T = 128
dl = DataLoaderLite(B, T, "datasets/shake.txt")

# --------- TRAINING ---------


@TinyJit
def train_step(x, y):
    optim.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optim.step()
    return loss.realize()


for step in (t := trange(max_steps)):
    t0 = perf_counter_ns()
    x, y = dl.next_batch()
    lr = get_lr(step)
    for opt in optim.optimizers:
        opt.lr = lr
    with Tensor.train():
        loss = train_step(
            Tensor(x, dtype=dtypes.long), Tensor(y, dtype=dtypes.long)
        ).item()
    if step % 100 == 0:
        model.generate("Lucy went to the store, and she")
    dt = (perf_counter_ns() - t0) * 1e-6
    t.set_description(
        f"train loss: {loss:.2f} | dt: {dt:.2f} | tok/s {(dl.B*dl.T)/(dt*1e-3):.2f} | lr: {lr.item():.5f}"
    )
