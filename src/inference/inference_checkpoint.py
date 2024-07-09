import argparse

import tiktoken

from model.config import GPT2Large
from model.model import GPT2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GPT2 Inference From Checkpoint",
        description="Loads checkpointed weights and generates text.",
    )
    parser.add_argument(
        "checkpoint_dir", help="Input checkpoint dir (relative to checkpoints/)"
    )
    parser.add_argument("prompt", help="Input prompt for text generation")
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=2,
        help="Number of return sequences (default: 2)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length of generated text (default: 100)",
    )

    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    model = GPT2.load_checkpoint(args.checkpoint_dir)

    generated_text = model.generate(
        args.prompt,
        enc,
        num_return_sequences=args.num_return_sequences,
        max_length=args.max_length,
    )

    for text in generated_text:
        print(">", text)
