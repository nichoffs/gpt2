import math
import os
from dataclasses import dataclass
from sys import exit
from time import perf_counter_ns

import numpy as np
import tiktoken
from tinygrad import Device, Tensor, TinyJit, dtypes
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
        return Tensor(input_, device=device), Tensor(ind, device=device)
    if largest:
        input_ *= -1
    ind_part = np.argsort(input_, axis=dim)
    ind = np.take_along_axis(ind, ind_part, axis=dim)
    if largest:
        input_ *= -1
    val = np.take_along_axis(input_, ind_part, axis=dim)
    return Tensor(val, device=device), Tensor(ind, device=device)


# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
val_steps = 20


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
    def __init__(self, config: GPT2Config = GPT2Small, GPUS=f"{Device.DEFAULT}:0"):
        self.config = config

        self.GPUS = GPUS
        self.NUM_GPUS = len(self.GPUS)

        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, config.norm_eps)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        self.pos = Tensor.arange(
            0, config.block_size, dtype=dtypes.long, requires_grad=False
        )  # (T,)

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
        pos_emb = self.wpe(self.pos.shrink(((0, T), None)))  # (T,) -> (T,C)
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

    def generate(self, enc, num_return_sequences, max_length):
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = Tensor(tokens, dtype=dtypes.long, device=GPUS)
        xgen = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        with Tensor.inference_mode():
            while xgen.shape[1] < max_length:
                logits = self(xgen)[0][:, -1, :]
                probs = logits.softmax(-1)
                topk_probs, topk_indices = topk(probs, 50, GPUS, -1)
                ix = topk_probs.multinomial(1)
                xcol = topk_indices.gather(-1, ix)
                xgen = xgen.cat(xcol, dim=1)
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"Sample {i + 1}: {decoded}")

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


class FineWebDataLoaderLite:
    def __init__(self, B, T, file_path, split="train", GPUS=f"{Device.DEFAULT}:0"):
        self.B = B
        self.T = T

        self.GPUS = GPUS
        self.NUM_GPUS = len(GPUS)

        self.batch = lambda x: x.reshape(self.NUM_GPUS * B, T)

        enc = tiktoken.get_encoding("gpt2")

        data_root = "data"
        shards = os.listdir(data_root)
        shards = sorted([s for s in shards if split in s])
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = Tensor(npt, dtype=dtypes.long)
        return ptt

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[
            self.current_position : (self.current_position + B * T * self.NUM_GPUS) + 1
        ]
        x = self.batch(buf[:-1]).shard_(self.GPUS, axis=0)
        y = self.batch(buf[1:]).shard_(self.GPUS, axis=0)
        self.current_position += B * T * self.NUM_GPUS

        if self.current_position + (B * T * self.NUM_GPUS + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        return x, y


class ShakespeareDataLoaderLite:
    def __init__(self, B, T, file_path, GPUS=f"{Device.DEFAULT}:0"):
        self.B = B
        self.T = T

        self.GPUS = GPUS
        self.NUM_GPUS = len(GPUS)

        self.batch = lambda x: x.reshape(self.NUM_GPUS * B, T)

        with open(file_path, "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")

        tokens = enc.encode(text)
        self.tokens = Tensor(tokens, dtype=dtypes.long)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[
            self.current_position : (self.current_position + B * T * self.NUM_GPUS + 1)
        ]
        x = self.batch(buf[:-1]).shard_(self.GPUS, axis=0)
        y = self.batch(buf[1:]).shard_(self.GPUS, axis=0)
        self.current_position += B * T * self.NUM_GPUS

        if self.current_position + (B * T * self.NUM_GPUS + 1) > len(self.tokens):
            print("read entire document, resetting position...")
            self.current_position = 0

        return x, y


# --------- INITIALIZATION ---------

GPUS = [f"{Device.DEFAULT}:{i}" for i in range(1)]
NUM_GPUS = len(GPUS)

total_batch_size = 2**9
B = 4
T = 32
assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by B*T"
grad_accum_steps = total_batch_size // (B * T * NUM_GPUS)
print(f"full batch size: {total_batch_size}, grad_accum_steps: {grad_accum_steps}")

# train_dl = FineWebDataLoaderLite(B, T, "datasets/shake.txt", "train", GPUS)
# val_dl = FineWebDataLoaderLite(B, T, "datasets/shake.txt", "train", GPUS)
train_dl = ShakespeareDataLoaderLite(B, T, "datasets/shake.txt", GPUS)
val_dl = ShakespeareDataLoaderLite(B, T, "datasets/shake.txt", GPUS)
model = GPT2(GPT2Small, GPUS)
for k, x in get_state_dict(model).items():
    x.to_(GPUS)
optim = configure_optimizers(get_parameters(model), 3e-4, 0.9, 0.95, 1e-8, 0.1)

# --------- TRAINING ---------


@TinyJit
def train_step(x, y):
    logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss.backward()
    for p in optim.params:
        p.grad.realize()
    return loss.realize()


@TinyJit
def val_step(x, y):
    logits, loss = model(x, y)
    for p in optim.params:
        p.grad.realize()
    return loss.realize()


@TinyJit
def optim_step():
    global_norm = Tensor([0.0], dtype=dtypes.float32, device=optim[0].device).realize()
    for p in optim.params:
        global_norm += p.grad.float().square().sum()
    global_norm = global_norm.sqrt()
    for p in optim.params:
        p.grad = (p.grad / Tensor.where(global_norm > 1.0, global_norm, 1.0)).cast(
            p.grad.dtype
        )

    optim.step()
    for p in opt.params:
        p.grad.assign(Tensor.zeros_like(p))
        p.grad.realize()


def full_train_step():
    with Tensor.train():
        full_batch_loss = 0.0
        for i in range(grad_accum_steps):
            x, y = train_dl.next_batch()
            loss = train_step(x, y)
            full_batch_loss += loss.item()
        optim_step()
        return full_batch_loss


def full_val_step():
    with Tensor.inference_mode():
        full_batch_loss = 0.0
        for i in range(val_steps):
            x, y = val_dl.next_batch()
            loss = val_step(x, y) / val_steps
            full_batch_loss += loss.item()
        return full_batch_loss


val_loss = None
enc = tiktoken.get_encoding("gpt2")
for step in (t := trange(max_steps)):
    t0 = perf_counter_ns()
    lr = Tensor([get_lr(step)], device=GPUS, requires_grad=False)
    for opt in optim.optimizers:
        opt.lr.assign(lr)
    loss = full_train_step()
    if not (step % 100) and step > 0:
        val_loss = full_val_step()
    if not step % 20:
        model.generate(enc, 2, 32)
    dt = (perf_counter_ns() - t0) * 1e-6
    t.set_description(
        f"train loss: {loss:.2f} | prev val loss: {val_loss if val_loss else 'None yet'} dt: {dt:.2f} | tok/s {(train_dl.B*train_dl.T*NUM_GPUS)/(dt*1e-3):.2f} | lr: {lr.item():.5f}"
    )
