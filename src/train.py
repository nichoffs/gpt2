import argparse
import os
from datetime import datetime
from time import perf_counter_ns

import tiktoken
from tinygrad import Tensor, TinyJit, dtypes
from tqdm import trange

from data.dataloader import ShardedDataLoaderLite
from model.config import TinyStories
from model.model import GPT2
from model.utils import get_lr, write_generations


def main(checkpoint_dir, save_dir, steps):
    os.makedirs("./generations", exist_ok=True)

    model_config = TinyStories()
    model = (
        GPT2(model_config)
        if not checkpoint_dir
        else GPT2.load_checkpoint(checkpoint_dir)
    )
    optim = model.configure_optimizers(lr=3e-4, b1=0.9, b2=0.95, eps=1e-8, wd=0.1)

    B = 12
    T = model_config.block_size
    train_dl = ShardedDataLoaderLite(B, T, "datasets/tinystories", "train")
    val_dl = ShardedDataLoaderLite(B, T, "datasets/tinystories", "val")

    encoder = tiktoken.get_encoding("gpt2")

    val_period = 500
    generate_period = 100
    checkpoint_period = 10
    if steps < checkpoint_period:
        print("WARNING: THIS RUN WILL NOT CHECKPOINT")

    @TinyJit
    def train_step(x, y):
        with Tensor.train():
            optim.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optim.step()
            return loss.realize()

    for step in (t := trange(steps)):
        t0 = perf_counter_ns()

        # load data and change buffer so JIT works
        x, y = train_dl.next_batch()
        x, y = Tensor(x.numpy(), dtype=dtypes.long), Tensor(
            y.numpy(), dtype=dtypes.long
        )

        # configure learning rate
        lr = get_lr(step, steps)
        for opt in optim.optimizers:
            opt.lr.assign(lr)

        # forward, backward, and step
        loss = train_step(x, y).item()

        # validate
        if step % val_period == 0:
            val_loss_accum = 0
            for _ in range(10):
                x, y = val_dl.next_batch()
                with Tensor.inference_mode():
                    logits, val_loss = model(x, y)
                    val_loss_accum += val_loss.item()
            val_loss_accum /= 10

        if step % generate_period == 0:
            # generate some text
            sequences = model.generate("Lucy went to the store, and she ", encoder)
            write_generations(step, sequences)

        if step % checkpoint_period == 0 and step > 0:
            # save by time and step
            current_time = datetime.now().strftime("%b%d%I%M%p")
            checkpoint_dir = f"{current_time}/{step}"
            model.save_checkpoint(save_dir + "/" + checkpoint_dir)

        dt = (perf_counter_ns() - t0) * 1e-6
        t.set_description(
            f"train loss: {loss:.2f} | val loss: {val_loss.item()} | dt: {dt:.2f} | tok/s {(train_dl.B*train_dl.T)/(dt*1e-3):.2f} | lr: {lr.item():.5f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="training", description="training script based on TinyStories GPT2"
    )
    parser.add_argument("-c", "--checkpoint_load_dir", default="", required=False)
    parser.add_argument("-p", "--checkpoint_save_dir", default="", required=False)
    parser.add_argument("-s", "--steps", default="", required=False)
    args = parser.parse_args()

    main(args.checkpoint_load_dir, args.checkpoint_save_dir, int(args.steps))
