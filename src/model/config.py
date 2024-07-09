from dataclasses import dataclass


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    norm_eps: float = 1e-5


@dataclass
class TinyStories(GPT2Config):
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
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
