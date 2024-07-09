import argparse
import os

import numpy as np
import tiktoken
from tqdm import tqdm

from datasets import load_dataset


def tokenize(doc, enc, eot):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def process_split(split, enc, eot, DATA_CACHE_DIR, shard_size):
    fw = load_dataset("roneneldan/TinyStories", split=split)
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for doc in tqdm(fw, desc=f"Processing {split} documents"):
        tokens = tokenize(doc, enc, eot)
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size,
                    unit="tokens",
                    desc=f"{split.capitalize()} Shard {shard_index}",
                )
            progress_bar.update(len(tokens))
        else:
            filename = os.path.join(
                DATA_CACHE_DIR, f"tinystory_{split}_{shard_index:06d}"
            )
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
    if token_count != 0:
        filename = os.path.join(DATA_CACHE_DIR, f"tinystory_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])


def main():
    parser = argparse.ArgumentParser(description="Process TinyStories dataset.")
    parser.add_argument(
        "output_path", type=str, help="Path where the dataset will be saved"
    )
    args = parser.parse_args()

    shard_size = int(1e8)  # 100M tokens per shard
    DATA_CACHE_DIR = args.output_path
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    # Process train split
    process_split("train", enc, eot, DATA_CACHE_DIR, shard_size)
    # Process validation split
    process_split("validation", enc, eot, DATA_CACHE_DIR, shard_size)


if __name__ == "__main__":
    main()
