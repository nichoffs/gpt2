import os
from dataclasses import asdict
from json import dump, dumps, load, loads

import numpy as np
from tiktoken import get_encoding
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import fetch
from tinygrad.nn.optim import AdamW, OptimizerGroup
from tinygrad.nn.state import (
    get_parameters,
    get_state_dict,
    load_state_dict,
    torch_load,
)

from model.config import *
from model.utils import topk


class MLP:
    def __init__(self, config: GPT2Config):
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
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
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
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
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
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

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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

        return logits[:, -1, :]

    def generate(
        self, prompt, enc=get_encoding("gpt2"), num_return_sequences=2, max_length=30
    ):
        tokens = enc.encode(prompt)
        tokens = Tensor(tokens, dtype=dtypes.long)
        xgen = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        with Tensor.inference_mode():
            while xgen.shape[1] < max_length:
                logits = self(xgen)
                probs = logits.softmax(-1)
                topk_probs, topk_indices = topk(probs, 50, -1)
                ix = topk_probs.multinomial(1)
                xcol = topk_indices.gather(-1, ix)
                xgen = xgen.cat(xcol, dim=1)
            ret_seqs = []
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                ret_seqs.append(decoded)
            return ret_seqs

    def configure_optimizers(self, lr, b1, b2, eps, wd):
        # TODO: do I need to include requires_grad for the count to be correct?
        # Think about this when adding bias for attention

        parameters = get_parameters(self)

        params_nodecay = [
            param
            for param in parameters
            if len(param.shape) < 2
            and (param.requires_grad or param.requires_grad is None)
        ]
        params_decay = [
            param
            for param in parameters
            if len(param.shape) >= 2
            and (param.requires_grad or param.requires_grad is None)
        ]

        opt_decay = AdamW(params_decay, lr=lr, b1=b1, b2=b2, eps=eps, weight_decay=wd)
        opt_nodecay = AdamW(
            params_nodecay, lr=lr, b1=b1, b2=b2, eps=eps, weight_decay=0
        )

        num_params_decay = sum(param.numel() for param in opt_decay.params)
        num_params_nodecay = sum(param.numel() for param in opt_nodecay.params)

        print(
            f"num decay params: {num_params_decay} num nodecay params: {num_params_nodecay}"
        )

        optim_group = OptimizerGroup(opt_decay, opt_nodecay)
        return optim_group

    @staticmethod
    def load_pretrained(MODEL_NAME):
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

    @staticmethod
    def load_checkpoint(checkpoint_dir):
        with open(f"checkpoints/{checkpoint_dir}/config.json", "r") as f:
            config_dict = loads(load(f))
            config = GPT2Config(**config_dict)
        model = GPT2(config)
        with open(f"{checkpoint_dir}/model.npy", "rb") as f:
            for par in get_parameters(model):
                np_arr = np.load(f)
                """
                TODO: For some reason, for the first two params in the model,
                the network is saving an extra npy. Both happen to start
                with a 1, so I'm just skipping over them here.

                Don't feel like resolving this now and this works, so oh well.
                """
                if np_arr.shape[0] == 1:
                    np_arr = np.load(f)
                par.assign(Tensor(np_arr, dtype=dtypes.float32))
        return model

    def save_checkpoint(self, checkpoint_dir):
        os.makedirs(f"{checkpoint_dir}", exist_ok=True)
        with open(f"{checkpoint_dir}/model.npy", "wb") as f:
            for par in get_parameters(self):
                np.save(f, par.numpy())
        with open(f"{checkpoint_dir}/config.json", "w") as f:
            config_dict = {
                k: getattr(self.config, k)
                for k in dir(self.config)
                if not k.startswith("__")
            }
            config_json = dumps(config_dict)
            dump(config_json, f)

    def init_weights(self, param):
        if isinstance(param, nn.Linear):
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
        elif isinstance(param, nn.Embedding):
            param.weight = Tensor.normal(param.weight.shape, mean=0, std=0.02)
