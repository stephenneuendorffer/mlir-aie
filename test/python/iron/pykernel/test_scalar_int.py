# vector_vector_add/vector_vector_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates

import argparse
import sys
import numpy as np
import aie.iron as iron
from numpy.typing import *
from typing import *

from aie.iron import ObjectFifo, Program, Runtime, Worker, PyKernel
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import *
from aie.dialects import memref, arith
import aie.extras.dialects.ext.arith as arith_extras
import aie.extras.dialects.ext.scf as scf
from aie.helpers.util import np_dtype_to_mlir_type, infer_mlir_type
from aie.extras.runtime.passes import Pipeline
from aie.passmanager import PassManager
from aie.execution_engine import ExecutionEngine
import aie.dialects.func as func
import aie.dialects.tensor as tensor
import aie.dialects.index as index
import aie.ir

    
def test_model(fn, input0, params, output):
    num_elements = np.size(input0)
    offset = params[0]
    n = 64
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n
    dtype = input0.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(1, num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(1, n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_params = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    test_kernel = PyKernel(fn)

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_params, of_out, kernel):
        elem_in1 = of_in1.acquire(1)
        elem_params = of_params.acquire(1)
        elem_out = of_out.acquire(1)
    
        elem_out[0,0] = kernel(elem_in1[0,0])
        elem_out[0,1] = kernel(1)
        elem_out[0,2] = kernel(2)

        of_in1.release(1)
        of_params.release(1)
        of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in1.cons(), of_params.cons(), of_out.prod(), test_kernel])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_params.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())

# JIT-compile the kernel then launches the kernel with the given arguments. Future calls
# to the kernel will use the same compiled kernel and loaded code objects
def test_nop(x:int):
    return x

def test_mul(x:int):
    return x*2

def test_div(x:int):
    return x/2

def test_add(x:int):
    return x+2

def test_sub(x:int):
    return x-2

def test_fn(x:int):
    acc = 1
    for i in range(5):
        acc = acc + x
    return acc

def test_fn2(x:int):
    r = np.zeros((64, 64), int)
    # r[0,0] = 1
    return r

def test_fn3(x:int):
    return np.ndarray((x, x), int)

def test_fn7(r:array[int]):
    acc = 1
    for i in range(0,10):
        r[0,i] = i*2
    return r

def main():
    device_map = {
        "npu": NPU1Col1(),
        "npu2": NPU2Col1(),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["npu", "npu2"],
        default="npu",
        help="Target device",
    )
    parser.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=1024,
        help="Number of elements (default: 1024)",
    )
    args = parser.parse_args()

    # Construct two input random tensors and an output zeroed tensor
    # The three tensor are in memory accessible to the NPU
    input0 = iron.zeros((args.num_elements,), dtype=np.int32, device="npu")
    params = iron.randint(0, 1, (16,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input0)

    iron.set_current_device(device_map[args.device])

    import time
    def jit_test(test, result):
        iron.jit(partial(test_model, test), is_placed=False, use_cache=True)(input0, params, output)
        if str(output) != result:
            print(test, ": Got", str(output), "but expected", result)

    jit_test(test_nop, "tensor([0,1,2,...,0,0,0], device='npu')")
    jit_test(test_mul, "tensor([0,2,4,...,0,0,0], device='npu')")
    # jit_test(test_div, "")
    jit_test(test_add, "tensor([2,3,4,...,0,0,0], device='npu')")
    jit_test(test_sub, "tensor([-2,-1, 0,..., 0, 0, 0], device='npu')")


if __name__ == "__main__":
    main()
