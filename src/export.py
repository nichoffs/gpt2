from sys import exit

from tinygrad import Device, Tensor

from config import TinyStories
from export_model import export_model
from model import GPT2

Device.DEFAULT = "CLANG"
model = GPT2.load_pretrained("gpt2")
mode = "clang"
prg, inp_sizes, out_sizes, state = export_model(model, mode, Tensor.randn(1, 256))
print(type(prg))
print(inp_sizes.items())
print(out_sizes.items())
print(type(state))
