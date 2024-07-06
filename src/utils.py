import math

import numpy as np
from tinygrad import Tensor


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
        return Tensor(input_), Tensor(ind)
    if largest:
        input_ *= -1
    ind_part = np.argsort(input_, axis=dim)
    ind = np.take_along_axis(ind, ind_part, axis=dim)
    if largest:
        input_ *= -1
    val = np.take_along_axis(input_, ind_part, axis=dim)
    return Tensor(val), Tensor(ind)


# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 1000
max_steps = 2441405


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return Tensor([max_lr * (it + 1) / warmup_steps], requires_grad=False)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return Tensor([min_lr], requires_grad=False)
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return Tensor([min_lr + coeff * (max_lr - min_lr)], requires_grad=False)

def write_generations(step, generations):
    with open(f"./generations/step_{step}_generations.txt", "w") as f:
        for i, gen in enumerate(generations):
            f.write(f"STEP {step} SAMPLE {i}:\n\n")
            f.write(f"{gen}\n\n")
