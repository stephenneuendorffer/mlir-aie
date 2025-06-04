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

from aie.iron import ObjectFifo, Program, Runtime, Worker
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
import aie.dialects.func as func

import ast, inspect

import astroid
import astypes

node = astroid.extract_node('1 + 2.3')
t = astypes.get_type(node)
print(t.annotation)  # 'float'

from collections import defaultdict
import aie.extras.types as T

_ast_type_to_mlir_type = defaultdict(
    lambda: None,
    {
        # Signed integer types
        "int": T.i32,
        "float": T.f32,
    }
)

def to_mlir_type(x):
    mlir_type = _ast_type_to_mlir_type[x]
    if mlir_type:
        return mlir_type()
    else:
        raise AttributeError(
            f"Failed to map ast type to mlir python type: {str(x)}"
        )

# Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
#                  | RShift | BitOr | BitXor | BitAnd | FloorDiv

class IntegerOpEncoder(ast.NodeVisitor):
    def visit_Add(self, node):
        return arith.AddIOp

class FloatOpEncoder(ast.NodeVisitor):
    def visit_Add(self, node):
        return arith.AddFOp

def get_mlir_BinOp(op, type):
    if type == "int":
        return IntegerOpEncoder().visit(op)
    elif type == "float":
        return FloatOpEncoder().visit(op)
    else:
        raise Exception("Unknown binop type")

class CodeGenerator(ast.NodeVisitor):
    def __init__(self, typetree):
        self.indent = ""
        self.environment = {}
        self.typetree = typetree

    def generic_visit(self, node):
        print(self.indent, node)
        raise Exception("Unsupported python node", node)
        
    def visit_Module(self, node):
        # Traverse all the sub-nodes
        for child in node.body:
            ast.NodeVisitor.visit(self, child)        

    def visit_FunctionDef(self, node):
        print(node, astypes.get_type(astypes.find_node(self.typetree, node)))

        # Walk the arguments and find their type annotations
        argtypes = []
        argnames = []
        for arg in node.args.args:
            print(arg, astypes.get_type(astypes.find_node(self.typetree, arg)), to_mlir_type(arg.annotation.id))
            argtypes.append(to_mlir_type(arg.annotation.id))
            argnames.append(arg.arg)

        # Walk the return operations and infer their types.  hopefully they are all the same.
        returntype = None
        for opnode in node.body:
            if isinstance(opnode, ast.Return):
                print(opnode, astypes.get_type(astypes.find_node(self.typetree, opnode.value)))
                returntype = to_mlir_type(astypes.get_type(astypes.find_node(self.typetree, opnode.value))._name)

        print(returntype)
        foo = func.FuncOp("foo", (argtypes, [returntype]))

        #foo.sym_visibility = StringAttr.get("private")
        entry_block = foo.add_entry_block()
        inner_args = entry_block.arguments
        for (i, arg) in enumerate(argnames):
            self.environment[arg] = inner_args[i]

        with InsertionPoint(entry_block):
            # Traverse all the sub-nodes
            for child in node.body:
                ast.NodeVisitor.visit(self, child)

    def visit_Assign(self, node):
        value = self.visit(node.value)
        for target in node.targets:
            #self.fctx.update_loc(target)
            if not isinstance(target.ctx, ast.Store):
                # TODO: Del, AugStore, etc
                print("Unsupported assignment context type %s" %
                                target.ctx.__class__.__name__)
            self.environment[target.id] = value

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        lefttype = astypes.get_type(astypes.find_node(self.typetree, node.left))
        righttype = astypes.get_type(astypes.find_node(self.typetree, node.right))
        print(lefttype)
        print(righttype)

        # FIXME: handle promotion
        if lefttype._name != righttype._name:
            raise AttributeError(
                f"BinOp types don't match: {lefttype._name} and {righttype._name} in '{ast.unparse(node)}'"
            )
        return get_mlir_BinOp(node.op, lefttype._name)
        
    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            print("Unsupported expression name context type %s" %
                            node.ctx.__class__.__name__)
        
        return self.environment[node.id]

    def visit_For(self, node):
        (lb, ub, step) = (arith_extras.constant(0),arith_extras.constant(5),arith_extras.constant(1)) # FIXME
        iter_args = ["acc"] # FIXME: Walk the loop to figure this out.
        liveins = [self.environment[arg] for arg in iter_args]
        loop = scf.ForOp(lb, ub, step, liveins)
        with InsertionPoint(loop.body):
            self.environment[node.target.id] = loop.induction_variable
            for child in node.body:
                ast.NodeVisitor.visit(self, child)

            scf.YieldOp(loop.inner_iter_args)
        for (i, arg) in enumerate(iter_args):
            self.environment[arg] = loop.results[i]
        return loop

    def visit_Return(self, node):
        # add a terminator
        func.ReturnOp([self.visit(node.value)])

    def visit_Constant(self, node):
        return arith_extras.constant(node.value)

def test_fn(x:float):
    acc = 0.0
    for i in range(5):
        acc = acc + x
    return acc


with mlir_mod_ctx() as ctx:
    tree = ast.parse(inspect.getsource(test_fn))
    # node = astypes.find_node(tree, tree)
    # node_type = astypes.get_node(node)
    print(ast.dump(tree, indent=4))
    typetree = astroid.parse(inspect.getsource(test_fn))
    # print(next(tree.infer())) #.annotation)
    # print(ast.dump(tree, indent=4))

    generator = CodeGenerator(typetree)
    generator.visit(tree)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

# with mlir_mod_ctx() as ctx:
#     @device(AIEDevice.npu2_1col)
#     def device_body():
#         @core(tile(0, 2))
#         def mlir_test_fn():
#             acc = arith.ConstantOp(infer_mlir_type(0), 0)
#             for j in range_(5, iter_args=[acc], insert_yield=False):
#                 acc = j[1] + j[0]
#                 scf.yield_([acc])
#             return 
#     res = ctx.module.operation.verify()
#     if res == True:
#         print(ctx.module)
#     else:
#         print(res)

sys.exit(0)

@iron.jit(is_placed=False)
def vector_vector_add(input0, params, output):
    num_elements = np.size(input0)
    offset = params[0]
    n = 1024
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n
    dtype = input0.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_params = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_params, of_out):
        elem_in1 = of_in1.acquire(1)
        elem_params = of_params.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(num_elements):
            zero = arith.ConstantOp(infer_mlir_type(0), 0)
            elem_out[i] = arith.Scalar(zero)
            
        for i in range_(16):
            acc = arith.ConstantOp(infer_mlir_type(0), 0)
            for j in range_(num_elements, iter_args=[acc], insert_yield=False):
                acc = j[1] + elem_in1[i+j[0]] * elem_in1[j[0]]
                scf.yield_([acc])
            elem_out[i] = j[2]
            # arith.index_cast(i, to=np_dtype_to_mlir_type(dtype))
        of_in1.release(1)
        of_params.release(1)
        of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in1.cons(), of_params.cons(), of_out.prod()])

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
