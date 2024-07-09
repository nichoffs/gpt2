from config import GPT2Small, TinyStories
from model import GPT2

prompt = "Once upon a time"


def pretrain_gen():
    model_pretrained = GPT2.load_pretrained("gpt2")
    pretrained_gen = model_pretrained.generate(
        prompt, num_return_sequences=1, max_length=100
    )[0]
    return pretrained_gen


def checkpoint_gen():
    model_checkpoint = GPT2.load_checkpoint("Jul071039PM/10000")
    checkpoint_gen = model_checkpoint.generate(
        prompt, num_return_sequences=1, max_length=100
    )[0]
    return checkpoint_gen


with open("generation.txt", "w") as f:
    f.write(
        f"""
Pretrained Generation: {pretrain_gen()}
Checkpointed Generation: {checkpoint_gen()}
            """
    )
