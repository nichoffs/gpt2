# from src.config import GPT2Small, TinyStories
import os
import sys
from sys import exit
from textwrap import wrap

from model.config import TinyStories
from model.model import GPT2

prompt = "Once upon a time"


def pretrain_gen():
    model_pretrained = GPT2.load_pretrained("gpt2")
    pretrained_gen = model_pretrained.generate(
        prompt, num_return_sequences=1, max_length=50
    )[0]
    return wrap(pretrained_gen)


def checkpoint_gen():
    model_checkpoint = GPT2.load_checkpoint("checkpoints/FullRun/5000")
    checkpoint_gen = model_checkpoint.generate(
        prompt, num_return_sequences=1, max_length=50
    )[0]
    return wrap(checkpoint_gen)


with open("generation.txt", "w") as f:
    nl = "\n"
    f.write(
        f"""
Pretrained Generation: {nl.join(pretrain_gen())}
Checkpointed Generation: {nl.join(checkpoint_gen())}
            """
    )
