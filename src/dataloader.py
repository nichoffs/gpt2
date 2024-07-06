# TODO: change to native Tensor
import os

import numpy as np
import tiktoken
from tinygrad import Tensor, dtypes


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

class ShardedDataLoaderLite:
    def __init__(self, B, T, ds_dir, split="train"):
        self.B = B
        self.T = T

        self.batch = lambda x: x.reshape(B, T)

        enc = tiktoken.get_encoding("gpt2")

        data_root = ds_dir
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
            self.current_position : (self.current_position + B * T) + 1
        ]
        x = self.batch(buf[:-1])
        y = self.batch(buf[1:])
        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            print("loading next shard...")
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        return x, y
