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

import ast, inspect
import mypy.parse as mp

import astroid
import astypes

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)



class TypePrinter(ast.NodeVisitor):
    def __init__(self, typetree):
        self.indent = ""
        self.environment = {}
        self.typetree = typetree

    def generic_visit(self, node):
        print(self.indent, node)
        raise Exception("Unsupported python node", node)
        
    def visit_Call(self, node):
        print(self.indent, node)

    def visit_For(self, node):
        print("for")
        itertype = astypes.get_type(astypes.find_node(self.typetree, node.iter))
        print(ast.dump(node.iter), list(astypes.find_node(self.typetree, node.iter).infer()), itertype)
        # print(list(astypes.find_node(self.typetree, node.target).infer()))
        # Traverse all the sub-nodes
        for child in node.body:
            ast.NodeVisitor.visit(self, child) 

    def visit_Module(self, node):
        # Traverse all the sub-nodes
        for child in node.body:
            ast.NodeVisitor.visit(self, child)

    def visit_FunctionDef(self, node):
        print(node, astypes.find_node(self.typetree, node).args, astypes.get_type(astypes.find_node(self.typetree, node)))

        # Walk the arguments and find their type annotations
        argtypes = []
        argnames = []
        for arg in node.args.args:
            # print(arg, astypes.find_node(self.typetree, arg), astypes.get_type(astypes.find_node(self.typetree, arg)))
            argtypes.append((arg.arg, arg.annotation))
            # argnames.append(arg.arg)

        # Walk the return operations and infer their types.  hopefully they are all the same.
        returntype = None
        for opnode in node.body:
            if isinstance(opnode, ast.Return):
                if(opnode.value is not None):
                    # print(astypes.find_node(self.typetree, opnode))
                    inferred_type = astypes.get_type(astypes.find_node(self.typetree, opnode.value))
                    print(opnode, inferred_type)
                    if inferred_type:
                        returntype = inferred_type._name
                    else:
                        returntype = None

        print(node.name, argtypes, returntype)

        # Traverse all the sub-nodes
        for child in node.body:
            ast.NodeVisitor.visit(self, child)

    def visit_Assign(self, node):
        value = self.visit(node.value)
        righttype = astypes.get_type(astypes.find_node(self.typetree, node.value))
        for target in node.targets:
            #self.fctx.update_loc(target)
            if not isinstance(target.ctx, ast.Store):
                # TODO: Del, AugStore, etc
                print("Unsupported assignment context type %s" %
                                target.ctx.__class__.__name__)
            inferred_type = astypes.get_type(astypes.find_node(self.typetree, target))
            print("Assign", ast.unparse(target), inferred_type, "=", righttype)
            # self.environment[target.id] = value

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        lefttype = astypes.get_type(astypes.find_node(self.typetree, node.left))
        righttype = astypes.get_type(astypes.find_node(self.typetree, node.right))
        mytype = astypes.get_type(astypes.find_node(self.typetree, node))
        print("BinOp", mytype._name, "=", lefttype._name, righttype._name)
        
    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            print("Unsupported expression name context type %s" %
                            node.ctx.__class__.__name__)
        
        # return self.environment[node.id]


    def visit_Return(self, node):
        None

    def visit_Constant(self, node):
        None

def process_core_function(fn):
        tree = ast.parse(inspect.getsource(fn))
        typetree = astroid.parse(inspect.getsource(fn))
        node = tree.body[0]
        print(ast.dump(node, indent=4))
        print(node, astypes.get_type(astypes.find_node(typetree.body[0], node)))            

        generator = TypePrinter(typetree)
        generator.visit(tree.body[0])

def test_fn(x:int):
    acc = 1
    for i in [0,1,2,3,4]: #range(5):
        acc = acc + x
    return acc

def test_fn2(x:int):
    r = np.ndarray((64, 64), float)
    r[0,0] = 1.0
    return r

def test_fn3(x:int):
    return np.ndarray((x, x), float)

def test_fn4(r:array[int]):
    acc = 1
    for i in range(0,10):
        r[0,i] = i*2
    return r
    
# process_core_function(test_fn)
# process_core_function(test_fn2)
# process_core_function(test_fn3)
# process_core_function(test_fn4)
    # def process_numpy_array(data: NDArray[np.int_]) -> NDArray[np.float_]:
    # return data * 2.5


@iron.jit(is_placed=False)
def vector_vector_add(input0, params, output):
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

    test_kernel = PyKernel(test_fn4)

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_params, of_out, kernel):
        elem_in1 = of_in1.acquire(1)
        elem_params = of_params.acquire(1)
        elem_out = of_out.acquire(1)
        #elem_out[0] = 
        kernel(elem_out)
        # for i in range_(num_elements):
        #     zero = arith.ConstantOp(infer_mlir_type(0), 0)
        #     elem_out[i] = arith_extras.Scalar(zero)
            
        # for i in range_(16):
        #     acc = arith.ConstantOp(infer_mlir_type(0), 0)
        #     for j in range_(num_elements, iter_args=[acc], insert_yield=False):
        #         acc = j[1] + elem_in1[i+j[0]] * elem_in1[j[0]]
        #         scf.yield_([acc])
        #     elem_out[i] = j[2]
        #     # arith.index_cast(i, to=np_dtype_to_mlir_type(dtype))
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
    input0 = iron.randint(0, 20, (args.num_elements,), dtype=np.int32, device="npu")
    params = iron.randint(0, 1, (16,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input0)

    iron.set_current_device(device_map[args.device])

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    vector_vector_add(input0, params, output)

    print(output)

    # # Check the correctness of the result
    # e = np.equal(input0.numpy() + input1.numpy(), output.numpy())
    # errors = np.size(e) - np.count_nonzero(e)

    # # Optionally, print the results
    # if args.verbose:
    #     print(f"{'input0':>4} + {'input1':>4} = {'output':>4}")
    #     print("-" * 34)
    #     count = input0.numel()
    #     for idx, (a, b, c) in enumerate(
    #         zip(input0[:count], input1[:count], output[:count])
    #     ):
    #         print(f"{idx:2}: {a:4} + {b:4} = {c:4}")

    # # If the result is correct, exit with a success code.
    # # Otherwise, exit with a failure code
    # if not errors:
    #     print("\nPASS!\n")
    #     sys.exit(0)
    # else:
    #     print("\nError count: ", errors)
    #     print("\nFailed.\n")
    #     sys.exit(-1)


if __name__ == "__main__":
    main()
