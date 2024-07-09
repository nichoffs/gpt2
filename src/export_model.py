import json
from typing import Dict, List, Tuple

from tinygrad.dtype import DType, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context
from tinygrad.nn.state import get_state_dict
from tinygrad.renderer import Program
from tinygrad.tensor import Device, Tensor

EXPORT_SUPPORTED_DEVICE = ["WEBGPU", "WEBGL", "CLANG", "CUDA", "GPU"]


def export_model(model, target: str, *inputs):
    assert (
        Device.DEFAULT in EXPORT_SUPPORTED_DEVICE
    ), "only WEBGPU, WEBGL, CLANG, CUDA, GPU, METAL are supported"
    with Context(JIT=2):
        run, special_names = jit_model(model, *inputs)
    functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
    state = get_state_dict(model)
    weight_names = {id(x.lazydata.base.realized): name for name, x in state.items()}
    input_names = [name for _, name in special_names.items() if "input" in name]
    output_names = [name for _, name in special_names.items() if "output" in name]
    prg = ""
    if target == "clang":
        prg = export_model_clang(
            functions, statements, bufs, bufs_to_save, input_names, output_names
        )
    return (
        prg,
        {input: bufs[input][0] for input in input_names},
        {output: bufs[output][0] for output in output_names},
        state,
    )


def export_model_clang(
    functions: Dict[str, str],
    statements: Dict[str, Tuple[str, int, int]],
    bufs: Dict[str, Tuple[str, int, int]],
    bufs_to_save: Dict[str, Tensor],
    input_names: List[str],
    output_names: List[str],
) -> str:
    cprog = ["#include <tgmath.h>"]

    for name, cl in bufs_to_save.items():
        weight = "".join(["\\x%02X" % x for x in bytes(cl._buf)])
        cprog.append(f'unsigned char {name}_data[] = "{weight}";')

    inputs = ", ".join([f"float* {input}" for input in input_names])
    outputs = ", ".join([f"float* {output}" for output in output_names])
    cprog += [
        f"float {name}[{len}];"
        if name not in bufs_to_save
        else f"float *{name} = (float *){name}_data;"
        for name, (len, dtype, _key) in bufs.items()
        if name not in ["input", "outputs"]
    ]
    cprog += list(functions.values())
    cprog += (
        [f"void net({inputs}, {outputs}) {{"]
        + [
            f"{name}({', '.join(args)});"
            for (name, args, _global_size, _local_size) in statements
        ]
        + ["}"]
    )
    return "\n".join(cprog)


def compile_net(
    run: TinyJit, special_names: Dict[int, str]
) -> Tuple[
    Dict[str, str],
    List[Tuple[str, List[str], List[int]]],
    Dict[str, Tuple[int, DType, int]],
    Dict[str, Tensor],
]:
    functions, bufs, bufs_to_save, statements, bufnum = {}, {}, {}, [], 0
    for ji in run.jit_cache:
        fxn: Program = ji.prg.p
        functions[
            fxn.function_name
        ] = fxn.src  # NOTE: this assumes all with the same name are the same
        cargs = []
        for i, arg in enumerate(ji.bufs):
            key = id(arg)
            if key not in bufs:
                if key in special_names:
                    bufs[key] = (
                        special_names[key],
                        arg.size * arg.dtype.itemsize,
                        arg.dtype,
                        key,
                    )
                else:
                    bufs[key] = (
                        f"buf_{bufnum}",
                        arg.size * arg.dtype.itemsize,
                        arg.dtype,
                        key,
                    )
                    bufnum += 1
                    if i > 0:
                        bufs_to_save[
                            bufs[key][0]
                        ] = arg  # if first usage of a buffer is not an output, and it's not a special name
            cargs.append(bufs[key][0])
        statements.append((fxn.function_name, cargs, fxn.global_size, fxn.local_size))

    return (
        functions,
        statements,
        {name: (size, dtype, key) for (name, size, dtype, key) in bufs.values()},
        bufs_to_save,
    )


def jit_model(model, *args) -> Tuple[TinyJit, Dict[int, str]]:
    assert hasattr(model, "forward") or callable(
        model
    ), "model needs a forward function"

    @TinyJit
    def run(*x):
        out = model.forward(*x) if hasattr(model, "forward") else model(*x)
        assert (
            isinstance(out, tuple) or isinstance(out, list) or isinstance(out, Tensor)
        ), "model output must be a Tensor, tuple, or a list of Tensors for export"
        out = [out] if isinstance(out, Tensor) else out
        return [o.realize() for o in out]

    # twice to run the JIT
    for _ in range(2):
        the_output = run(*args)
    special_names = {}

    # hack to put the inputs back
    for (j, i), idx in run.input_replace.items():
        realized_input = args[idx].lazydata.base.realized
        run.jit_cache[j].bufs[i] = realized_input
        special_names[id(realized_input)] = f"input{idx}"

    # TODO: fetch this from the jit in self.input_replace and self.ret (hint: use get_parameters on self.ret)
    for i, output in enumerate(the_output):
        special_names[id(output.lazydata.base.realized)] = f"output{i}"
    return run, special_names
