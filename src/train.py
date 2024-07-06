import tiktoken
from tinygrad import Tensor, TinyJit, dtypes
from tqdm import trange
from time import perf_counter_ns
import os

from config import TinyStories
from gpt2 import GPT2
from utils import get_lr, write_generations
from dataloader import ShardedDataLoaderLite

# Create the generations directory if it doesn't exist
os.makedirs("./generations", exist_ok=True)

model_config = TinyStories()
model = GPT2(model_config)
optim = model.configure_optimizers(lr=3e-4, b1=.9, b2=.95, eps=1e-8, wd=0.1)
B = 16
T = model_config.block_size
train_dl = ShardedDataLoaderLite(B, T, "datasets/tinystories", "train")
val_dl = ShardedDataLoaderLite(B, T, "datasets/tinystories", "val")

encoder = tiktoken.get_encoding("gpt2")

max_steps = 1000
val_period = 50
num_samples = 2
max_new_tokens = 30

@TinyJit
def train_step(x, y):
    with Tensor.train():
        optim.zero_grad()
        logits, loss = model(x,y)
        loss.backward()
        optim.step()
        return loss.realize()

for step in (t := trange(max_steps)):
    t0 = perf_counter_ns()
    x, y = train_dl.next_batch()
    x, y = Tensor(x.numpy(), dtype=dtypes.long), Tensor(y.numpy(), dtype=dtypes.long)
    lr = get_lr(step)
    for opt in optim.optimizers:
        opt.lr.assign(lr)
    loss = train_step(x, y).item()
    if step % val_period == 0:
        val_loss_accum = 0
        for _ in range(10):
            x, y = val_dl.next_batch()
            with Tensor.inference_mode():
                logits, val_loss = model(x,y)
                val_loss_accum += val_loss.item()
        val_loss_accum /= 10

        # Generate sequences and write to file
        prompt = "Lucy went to the store, and she "
        sequences = model.generate(prompt, encoder, num_samples, max_new_tokens)
        write_generations(step, sequences)

    dt = (perf_counter_ns() - t0) * 1e-6
    t.set_description(
        f"train loss: {loss:.2f} | val loss: {val_loss.item()} | dt: {dt:.2f} | tok/s {(train_dl.B*train_dl.T)/(dt*1e-3):.2f} | lr: {lr.item():.5f}"
    )

# TODO: Adjust lr
