# pykernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import argparse
import sys
import numpy as np

from ..dialects.aie import *
from ..dialects.aiex import *
from ..extras.context import mlir_mod_ctx
from ..helpers.dialects.ext.scf import *
from ..dialects import memref, arith, func, tensor, index, scf
from ..extras.dialects.ext.arith import constant
from ..extras.dialects.ext.memref import alloca
from ..helpers.util import np_dtype_to_mlir_type, infer_mlir_type
# from ..extras.runtime.passes import Pipeline
# from ..passmanager import PassManager
# from ..execution_engine import ExecutionEngine
from .resolvable import Resolvable
from .. import ir

import ast, inspect
import mypy.parse as mp

import astroid
import astypes

from collections import defaultdict
import aie.extras.types as T

import logging

_ast_type_to_mlir_type = defaultdict(
    lambda: None,
    {
        # Signed integer types
        "int": T.i32,
        "float": T.f32,
        "ndarray": T.tensor,
    }
)

def to_mlir_type(x):
    if x == "int":
        return T.i32()
    elif x == "float":
        return T.f32()
    # Handle AST nodes to deal with type annotations
    elif isinstance(x, ast.Name):
        return to_mlir_type(x.id)
    elif isinstance(x, ast.Subscript):
        element_type = to_mlir_type(x.slice)
        if x.value.id == "ndarray" or x.value.id == "array" or x.value.id == "Sequence":
            return T.memref(element_type)
        raise AttributeError(
            f"Failed to map ast type to mlir python type: {str(x)}"
        )
    # Handle Astroid types
    elif isinstance(x, astypes.Type):
        if x._name == "ndarray" or x._name == "array" or x._name == "Sequence":
            element_type = to_mlir_type(x._args[0])
            return T.memref(element_type)
        else:
            return to_mlir_type(x._name)
    else:
        raise AttributeError(
            f"Failed to map ast type to mlir python type: {str(x)}"
        )

# Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
#                  | RShift | BitOr | BitXor | BitAnd | FloorDiv

class IntegerOpEncoder(ast.NodeVisitor):
    def visit_Add(self, node):
        return arith.addi

    def visit_Sub(self, node):
        return arith.subi

    def visit_Mult(self, node):
        return arith.muli

    def visit_Div(self, node):
        return arith.divi

class FloatOpEncoder(ast.NodeVisitor):
    def visit_Add(self, node):
        return arith.addf

    def visit_Sub(self, node):
        return arith.subf

    def visit_Mult(self, node):
        return arith.mulf

    def visit_Div(self, node):
        return arith.divf

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

    def get_type(self, node):
        result = astypes.get_type(astypes.find_node(self.typetree, node))
        if result is None:

            print("Unknown Type for", ast.dump(node))
            raise Exception("Type Inference failure", node) 
        logging.debug(f"Type of {ast.dump(node)} is {result}")
        return result

    def generic_visit(self, node):
        print(self.indent, node)
        raise Exception("Unsupported python node", node)
        
    def visit_Tuple(self, node):
        return [ast.NodeVisitor.visit(self, x) for x in node.elts]

    def visit_Call(self, node):
        # print(self.indent, node)
        if(node.func.attr == 'ndarray'):
            args = [index.CastUOp(T.index(), ast.NodeVisitor.visit(self, x)) for x in node.args[0].elts]
            #op = memref.AllocOp(T.memref(T.f32()), args, [])
            allocaop = alloca(args, T.f32(), alignment=64)
            op = memref.CastOp(T.memref(ir.ShapedType.get_dynamic_size(), ir.ShapedType.get_dynamic_size(), T.f32()), alloca)
            return op
            #        return tensor.CastOp(to_mlir_type('ndarray'), op)

    def visit_Module(self, node):
        return [ast.NodeVisitor.visit(self, child) for child in node.body]

    def visit_FunctionDef(self, node):
        # Walk the arguments and find their type annotations
        argtypes = []
        argnames = []
        for arg in node.args.args:
            argtypes.append(to_mlir_type(arg.annotation))
            argnames.append(arg.arg)

        # Walk the return operations and infer their types.  hopefully they are all the same.
        returntypes = []
        for opnode in node.body:
            if isinstance(opnode, ast.Return):
                if(opnode.value is not None):
                    # print(astypes.find_node(self.typetree, opnode))
                    inferred_type = self.get_type(opnode.value)
                    returntype = to_mlir_type(inferred_type)
                    returntypes.append(returntype)

        newfunc = FuncOp(node.name, (argtypes, returntypes))
        newfunc.sym_visibility = ir.StringAttr.get("private")

        #foo.sym_visibility = StringAttr.get("private")
        entry_block = newfunc.add_entry_block()
        inner_args = entry_block.arguments
        for (i, arg) in enumerate(argnames):
            self.environment[arg] = inner_args[i]

        with InsertionPoint(entry_block):
            for child in node.body:
                ast.NodeVisitor.visit(self, child)
    
            # Deal with implicit return
            if not isinstance(node.body[-1], ast.Return):
                func.ReturnOp([])

        return newfunc

    def is_slice(self, node):
        if isinstance(node, ast.Tuple):
            for e in node.elts:
                if isinstance(e, ast.Slice):
                    return True
        else:
            if isinstance(node, ast.Slice):
                return True
        return False

    def get_slice_as_index(self, node):
        result = []
        if isinstance(node, ast.Tuple):
            for e in node.elts:
                result.extend(self.get_slice_as_index(e))
        else:
            if isinstance(node, ast.Slice):
                result.append([ast.NodeVisitor.visit(self, node.lower),
                            arith.SubIOp(ast.NodeVisitor.visit(self, node.upper), ast.NodeVisitor.visit(self, node.lower)),
                            constant(1)])
            else:
                result.append([ast.NodeVisitor.visit(self, node), constant(1), constant(1)])
        return result
    
    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            mem_op = self.visit(node.value)
            if self.is_slice(node.slice):
                slices = self.get_slice_as_index(node.slice)
                args = [[index.CastUOp(T.index(), x) for x in y] for y in zip(*slices)]            
                shape = [ir.ShapedType.get_dynamic_size() for x in slices]
                shaped_mem_op = memref.cast(T.memref(*shape, T.i32()), mem_op)
                return memref.subview(shaped_mem_op, *args,
                                      result_type = T.memref(*shape, T.i32(), layout=ir.StridedLayoutAttr.get(ir.ShapedType.get_dynamic_size(), shape)))
            else:
                indexes = self.visit(node.slice)
                # print(var, indexes)
                from collections.abc import Iterable
                if not isinstance(indexes, Iterable):
                    indexes = [indexes]
                args = [index.CastUOp(T.index(), x) for x in indexes]            
                shape = [ir.ShapedType.get_dynamic_size() for x in indexes]
                shaped_mem_op = memref.CastOp(T.memref(*shape, T.i32()), mem_op)
                return memref.load(shaped_mem_op, args)
        else:
            # TODO: Del, AugStore, etc
            print("Unsupported assignment context type %s" %
                            target.ctx.__class__.__name__)

    def _do_assign(self, target, value):
        #self.fctx.update_loc(target)
        if isinstance(target, ast.Name):
            if isinstance(target.ctx, ast.Store):
                self.environment[target.id] = value
            else:
                # TODO: Del, AugStore, etc
                print("Unsupported assignment context type %s" %
                                target.ctx.__class__.__name__)
        elif isinstance(target, ast.Subscript):
            if isinstance(target.ctx, ast.Store):
                if self.is_slice(target.slice):
                    mem_op = self.visit(target.value)
                    slices = self.get_slice_as_index(target.slice)
                    args = [[index.CastUOp(T.index(), x) for x in y] for y in zip(*slices)]            
                    shape = [ir.ShapedType.get_dynamic_size() for x in slices]
                    shaped_mem_op = memref.cast(T.memref(*shape, T.i32()), mem_op)
                    target_subview = memref.subview(shaped_mem_op, *args,
                                        result_type = T.memref(*shape, T.i32(), layout=ir.StridedLayoutAttr.get(ir.ShapedType.get_dynamic_size(), shape)))            
                    memref.copy(value, target_subview)
                else:
                    mem_op = self.visit(target.value)
                    indexes = self.visit(target.slice)
                    # print(var, indexes)
                    from collections.abc import Iterable
                    if not isinstance(indexes, Iterable):
                        indexes = [indexes]
                    args = [index.CastUOp(T.index(), x) for x in indexes]
                    shape = [ir.ShapedType.get_dynamic_size() for x in indexes]
                    shaped_mem_op = memref.CastOp(T.memref(*shape, T.i32()), mem_op)
                    memref.store(value, shaped_mem_op, args)
            else:
                # TODO: Del, AugStore, etc
                print("Unsupported assignment context type %s" %
                                target.ctx.__class__.__name__)
        else:
            # TODO: 
            print("Unsupported assignment target %s" %
                            target.__class__.__name__)

    def visit_AnnAssign(self, node):
        value = self.visit(node.value)
        self._do_assign(node.target, value)
    
    def visit_Assign(self, node):
        value = self.visit(node.value)
        for target in node.targets:
            self._do_assign(target, value)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        lefttype = self.get_type(node.left)
        righttype = self.get_type(node.right)
        #mytype = self.get_type(node)
        # print("BinOp")
        # print(mytype)
        # print(lefttype, list(astypes.find_node(self.typetree, node.left).infer()))
        # print(righttype, list(astypes.find_node(self.typetree, node.right).infer()))

        # FIXME: handle promotion        
        if lefttype._name != righttype._name:
            raise AttributeError(
                f"BinOp types don't match: {str(lefttype)} and {str(righttype)} in '{ast.unparse(node)}'"
            )
        mlirop = get_mlir_BinOp(node.op, lefttype._name)
        return mlirop(left, right)
        
    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            print("Unsupported expression name context type %s" %
                            node.ctx.__class__.__name__)
        
        return self.environment[node.id]

    # Given 'iter' node of a For loop, return a (lb, ub, step) triple
    def _get_for_range(self, iter_node):
        args = iter_node.args
        if len(args) == 1:
            return (constant(0), self.visit(args[0]), constant(1))
        elif len(args) == 2:
            return (self.visit(args[0]), self.visit(args[1]), constant(1))
        else:
            return (self.visit(args[0]), self.visit(args[1]), self.visit(args[2]))

    def _get_for_loop(self, iter_node, liveins):
        if isinstance(iter_node, ast.Call) and iter_node.func.id == "range":
            (lb, ub, step) = self._get_for_range(iter_node)
            loop = ForOp(lb, ub, step, liveins)
            return (loop, loop.induction_variable)
        elif isinstance(iter_node, ast.List):
            itertype = self.get_type(iter_node.elts[0])
            size = len(iter_node.elts)
            g = memref.alloca(T.memref(size, to_mlir_type(itertype)), [], [])
            for i, e in enumerate(iter_node.elts):
                value = self.visit_Constant(e)
                memref.store(value, g, [index.constant(i)])
            loop = ForOp(index.constant(0), index.constant(size), index.constant(1), liveins)
            with InsertionPoint(loop.body):
                val = memref.load(g, [loop.induction_variable])
            return (loop, val)
        else:
            print("Unsupported loop iterator %s" %
                ast.dump(iter_node))

    def visit_For(self, node):
        iter_args = ["acc"] # FIXME: Walk the loop to figure this out.
        liveins = [self.environment[arg] for arg in iter_args]
        (loop, iv) = self._get_for_loop(node.iter, liveins)
        # Inside the loop all of the loop carried variables take on their loop-carried values
        for (i, arg) in enumerate(iter_args):
            self.environment[arg] = loop.inner_iter_args[i]
        with InsertionPoint(loop.body):
            self.environment[node.target.id] = iv

            for child in node.body:
                ast.NodeVisitor.visit(self, child)

            # At the end of the loop, yield any new values of loop-carried variables
            scf.YieldOp([self.environment[arg] for arg in iter_args])
        for (i, arg) in enumerate(iter_args):
            self.environment[arg] = loop.results[i]
        return loop

    def visit_Return(self, node):
        # add a terminator
        if(node.value is not None):
            func.ReturnOp([self.visit(node.value)])
        else:
            func.ReturnOp([])

    def visit_Constant(self, node):
        return constant(node.value)

def process_core_function(fn):
    if isinstance(fn, str):
        source = fn
    else:
        source = inspect.getsource(fn)

    try:
        tree = ast.parse(source)
    except Exception as e:
        print("parsing failed: ", source)

    typetree = astroid.parse(source)
    generator = CodeGenerator(typetree)
    result = None
    try:
        result = generator.visit(tree.body[0])
    except Exception as e:
        print("In: ")
        print(ast.dump(tree, indent=4))
        raise e

    return result
    # res = ctx.module.operation.verify()
    # if res == True:
    #     print(ctx.module)
    # else:
    #     print(res)

    # LOWER_TO_LLVM_PIPELINE = (
    #     Pipeline()
    #     .canonicalize()
    #     .cse()
    #     .one_shot_bufferize()
    #     .buffer_results_to_out_params()
    #     .convert_vector_to_llvm()
    #     .expand_strided_metadata()
    #     .lower_affine()
    #     .convert_math_to_llvm()
    #     .convert_index_to_llvm()
    #     .arith_expand()
    #     .convert_arith_to_llvm()
    #     .finalize_memref_to_llvm()
    #     .convert_func_to_llvm(use_bare_ptr_memref_call_conv=True)
    #     .convert_cf_to_llvm()
    #     .canonicalize()
    #     .cse()
    # )

    # pm = PassManager.parse(str(LOWER_TO_LLVM_PIPELINE))
    # try:
    #     pm.run(ctx.module.operation)
    # except Exception as e:
    #     print("Error running pass pipeline: ", pass_pipeline, e)
    #     raise e

    # print(ctx.module)
    
    
    #await self.do_call(task, ["aie-translate", "--mlir-to-llvmir", file_opt_core, "-o", file_core_llvmir])

def get_mlir(fn):
    with mlir_mod_ctx() as ctx:
        return str(process_core_function(fn))

class PyKernel(Resolvable):
    def __init__(
        self,
        name: str,
    ) -> None:
        """A Kernel is an externally defined function that eventually resolves to a FuncOp. If it is called,
        a CallOp will be generated.

        Args:
            name (str): The name of the function
        """
        self._name = name
        self._op: FuncOp | None = None

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._op:
            self._op = process_core_function(self._name)

    def downcast_arg(self, x, t):
        if isinstance(t, T.MemRefType) or isinstance(t, T.UnrankedMemRefType):
            return memref.CastOp(t, x)
        else:
            return x

    def __call__(self, *args, **kwargs):
        if not self._op:
            raise ValueError("Need to resolve PyKernel before it can be called")
        # print("Generating call")
        # print(args)
        # print(self._op)
        # print(self._op.type.inputs[0])
        downcasted_args = [self.downcast_arg(x, t) for (x, t) in zip(args, self._op.type.inputs)]
        return call(self._op, downcasted_args, **kwargs)