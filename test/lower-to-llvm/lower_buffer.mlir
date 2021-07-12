//===- lower_buffer.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-llvm-lowering="tilecol=3 tilerow=3" %s | FileCheck --check-prefix=CHECK33 %s
// RUN: aie-opt --aie-llvm-lowering="tilecol=4 tilerow=3" %s | FileCheck --check-prefix=CHECK43 %s
// CHECK33-LABEL:  llvm.func @core33() {
// CHECK33:    %[[VAR1:.*]] = llvm.mlir.addressof @a : !llvm.ptr<array<4 x i32>>
// CHECK33:    %[[VAR5:.*]] = llvm.getelementptr %[[VAR1]][%{{.*}}, %{{.*}}] : (!llvm.ptr<array<4 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK33:    llvm.store %{{.*}}, %[[VAR5]] : !llvm.ptr<i32>
// CHECK33:  }
// CHECK43-LABEL:  llvm.func @core43() {
// CHECK43:    %[[VAR1:.*]] = llvm.mlir.addressof @a : !llvm.ptr<array<4 x i32>>
// CHECK43:    %[[VAR5:.*]] = llvm.getelementptr %[[VAR1]][%{{.*}}, %{{.*}}] : (!llvm.ptr<array<4 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK43:    %{{.*}} = llvm.load %[[VAR5]] : !llvm.ptr<i32>
// CHECK43:  }

module @codegen1 {
  %t33 = AIE.tile(3, 3)
  %a = AIE.buffer(%t33) { sym_name = "a" } : memref<4xi32>
  %core33 = AIE.core(%t33) {
    %0 = constant 0 : index
    %377 = constant 377 : i32
    memref.store %377, %a[%0] : memref<4xi32>
    AIE.end
  }
  %t34 = AIE.tile(4, 3)

  %core34 = AIE.core(%t34) {
    %0 = constant 0 : index
    %1 = memref.load %a[%0] : memref<4xi32>
//    AIE.debug(%1 : i32)
    AIE.end
  }
}
