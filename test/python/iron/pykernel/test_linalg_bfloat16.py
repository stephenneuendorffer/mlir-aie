# vector_vector_add/vector_vector_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates

import argparse
import sys
import ml_dtypes as np_ml
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
import aie.dialects.linalg as linalg
import aie.ir
    
def test_model(fn, input0, params, output):
    num_elements = np.size(input0)
    shape = np.shape(input0)
    offset = params[0]
    n = 64
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n
    dtype = input0.dtype

    # Define tensor types
    tensor_ty = np.ndarray[shape, np.dtype[dtype]]
    tile_ty = np.ndarray[shape, np.dtype[dtype]]

    # AIE-array data movement with object fifos
    a_dims = [(8,1), (8, 8)]
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_params = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")
    #, dims_to_stream=a_dims)
    
    # memA = of_in1.cons().forward(name="memA", dims_to_stream=a_dims)
    test_kernel = PyKernel(fn, Pipeline().canonicalize().convert_linalg_to_affine_loops())#.add_pass("affine-raise-from-memref").affine_super_vectorize("8", vectorize_reductions=True))

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_params, of_out, kernel):
        elem_in1 = of_in1.acquire(1)
        elem_params = of_params.acquire(1)
        elem_out = of_out.acquire(1)
    
        kernel(elem_in1, elem_out)

        of_in1.release(1)
        of_params.release(1)
        of_out.release(1)

    # Create a worker to run the task on a compute tile
    # dims_from_stream=a_dims
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
def test_loop_bfloat16(x:Sequence[np_ml.bfloat16], o:Sequence[np_ml.bfloat16]):
    acc = 0

    y = np_ml.bfloat16(0.0)
    # y = x[0,0]
    for i in range(0,8):
        o[0,i] = np_ml.bfloat16(x[0,i]) + y + np_ml.bfloat16(i)



# from aie.iron.pykernel import get_mlir
# print(get_mlir(test_loop_bfloat16))
# def test_fn3(x:Sequence[int], o:Sequence[int]):
#     return np.ndarray((x, x), int)

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
        default=8,
        help="Number of elements (default: 8)",
    )
    args = parser.parse_args()

    # Construct two input random tensors and an output zeroed tensor
    # The three tensor are in memory accessible to the NPU
    input0 = iron.zeros((args.num_elements, args.num_elements), dtype=np_ml.bfloat16, device="npu")
    params = iron.rand(16, dtype=np_ml.bfloat16)
    output = iron.zeros_like(input0)

    iron.set_current_device(device_map[args.device])

    input0[0:args.num_elements] = range(1,args.num_elements+1)

    import time
    def jit_test(test, result):
        if args.verbose:
            from aie.iron.pykernel import get_mlir
            print(get_mlir(test))
        iron.jit(partial(test_model, test), is_placed=False, use_cache=True)(input0, params, output)
        if np.array_equal(output, result):
            print("passed...")
            return
        elif str(output) != str(result):
            print(test, ": Got", str(output), "but expected", result)


    golden_output = np.zeros_like(input0)
    test_loop_bfloat16(np.array(input0), golden_output)
    jit_test(test_loop_bfloat16, golden_output)


if __name__ == "__main__":
    main()
