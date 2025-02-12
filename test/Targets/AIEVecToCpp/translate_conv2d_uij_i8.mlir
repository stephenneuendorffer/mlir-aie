// RUN: aie-translate --aievec-to-cpp %s -split-input-file | FileCheck %s

// CHECK-LABEL: void conv2d_0(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
func @conv2d_0(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<48xi8>, vector<64xi8>
  %1 = aievec.upd %arg1[%c32], %0 {index = 1 : i8, offset = 0 : si32} : memref<48xi8>, vector<64xi8>
  %c0_0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0_0 to %c16 step %c1 {
    %c1_1 = arith.constant 1 : index
    %2 = arith.addi %arg3, %c1_1 : index
    %c2 = arith.constant 2 : index
    %3 = arith.addi %arg3, %c2 : index
    %c0_2 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c16_3 = arith.constant 16 : index
    scf.for %arg4 = %c0_2 to %c256 step %c16_3 {
      %4 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<16x256xi8>, vector<16xi8>
      %5 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<32xi8>
      %6 = aievec.ups %4 {shift = 10 : i8} : vector<16xi8>, !aievec.acc<16xi48>
      %7 = aievec.mac %0, %5, %6 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %8 = aievec.mac %0, %5, %6 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %9 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<32xi8>
      %10 = aievec.mac %0, %9, %7 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %11 = aievec.mac %0, %9, %8 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %12 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<32xi8>
      %13 = aievec.mac %1, %12, %10 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %14 = aievec.mac %1, %12, %11 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %15 = aievec.srs %13 {shift = 10 : i8} : !aievec.acc<16xi48>, vector<16xi16>
      %16 = aievec.srs %14 {shift = 10 : i8} : !aievec.acc<16xi48>, vector<16xi16>
      %17 = aievec.concat %15, %16 : vector<16xi16>, vector<32xi16>
      %18 = aievec.select %17 {select = "0xcccccccc", xoffsets = "0x0c080400", xoffsets_hi = "0x0", xsquare = "0x1010", xstart = "0", yoffsets = "0x0c080400", yoffsets_hi = "0x0", ysquare = "0x1010", ystart = "4"} : vector<32xi16>, vector<32xi16>
      %19 = aievec.ext %18 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
      %20 = aievec.pack %19 : vector<16xi16>, vector<16xi8>
      vector.transfer_write %20, %arg2[%arg3, %arg4] : vector<16xi8>, memref<16x256xi8>
    }
  }
  return
}

//CHECK-NEXT:  size_t v4 = 32;
//CHECK-NEXT:  size_t v5 = 0;
//CHECK-NEXT:  v64int8 v6;
//CHECK-NEXT:  int8_t * restrict r_v6_v2 = v2;
//CHECK-NEXT:  v6 = upd_w(v6, 0, *(v32int8 *)(r_v6_v2 + v5));
//CHECK-NEXT:  v6 = upd_w(v6, 1, *(v32int8 *)(r_v6_v2 + v4));
//CHECK-NEXT:  size_t v7 = 0;
//CHECK-NEXT:  size_t v8 = 16;
//CHECK-NEXT:  size_t v9 = 1;
//CHECK-NEXT:  for (size_t v10 = v7; v10 < v8; v10 += v9)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  chess_loop_range(16, 16)
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v11 = 1;
//CHECK-NEXT:    size_t v12 = v10 + v11;
//CHECK-NEXT:    size_t v13 = 2;
//CHECK-NEXT:    size_t v14 = v10 + v13;
//CHECK-NEXT:    size_t v15 = 0;
//CHECK-NEXT:    size_t v16 = 256;
//CHECK-NEXT:    size_t v17 = 16;
//CHECK-NEXT:    for (size_t v18 = v15; v18 < v16; v18 += v17)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(16, 16)
//CHECK-NEXT:    {
//CHECK-NEXT:      v16int8 v19 = *(v16int8 *)(v3 + 256*v10+v18);
//CHECK-NEXT:      v32int8 v20 = *(v32int8 *)(v1 + 288*v10+v18);
//CHECK-NEXT:      v16acc48 v21 = ups(v19, 10);
//CHECK-NEXT:      v21 = mac16(v21, v6, 0, 0x00000000, 4, 0x1010, v20, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v21 = mac16(v21, v6, 0, 0x00000000, 4, 0x1010, v20, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v32int8 v22 = *(v32int8 *)(v1 + 288*v12+v18);
//CHECK-NEXT:      v21 = mac16(v21, v6, 16, 0x00000000, 4, 0x1010, v22, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v21 = mac16(v21, v6, 16, 0x00000000, 4, 0x1010, v22, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v32int8 v23 = *(v32int8 *)(v1 + 288*v14+v18);
//CHECK-NEXT:      v21 = mac16(v21, v6, 32, 0x00000000, 4, 0x1010, v23, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v21 = mac16(v21, v6, 32, 0x00000000, 4, 0x1010, v23, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v16int16 v24 = srs(v21, 10);
//CHECK-NEXT:      v16int16 v25 = srs(v21, 10);
//CHECK-NEXT:      v32int16 v26 = concat(v24, v25);
//CHECK-NEXT:      v32int16 v27 = select32(0xcccccccc, v26, 0, 0x0c080400, 0x0, 0x1010, 4, 0x0c080400, 0x0, 0x1010);
//CHECK-NEXT:      v16int16 v28 = ext_w(v27, 0);
//CHECK-NEXT:      v16int8 v29 = pack(v28);
//CHECK-NEXT:      *(v16int8 *)(v3 + 256*v10+v18) = v29;
//CHECK-NEXT:    }
//CHECK-NEXT:  }

// CHECK-LABEL: void conv2d_1(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
func @conv2d_1(%arg0: memref<18x288xi8>, %arg1: memref<48xi8>, %arg2: memref<16x256xi8>) {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = aievec.upd %arg1[%c0] {index = 0 : i8, offset = 0 : si32} : memref<48xi8>, vector<64xi8>
  %1 = aievec.upd %arg1[%c32], %0 {index = 1 : i8, offset = 0 : si32} : memref<48xi8>, vector<64xi8>
  %c0_0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0_0 to %c16 step %c1 {
    %c1_1 = arith.constant 1 : index
    %2 = arith.addi %arg3, %c1_1 : index
    %c2 = arith.constant 2 : index
    %3 = arith.addi %arg3, %c2 : index
    %c0_2 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c16_3 = arith.constant 16 : index
    scf.for %arg4 = %c0_2 to %c256 step %c16_3 {
      %4 = aievec.upd %arg0[%arg3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<32xi8>
      %5 = aievec.mul %0, %4 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %6 = aievec.mul %0, %4 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "0", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %7 = aievec.upd %arg0[%2, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<32xi8>
      %8 = aievec.mac %0, %7, %5 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %9 = aievec.mac %0, %7, %6 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "16", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %10 = aievec.upd %arg0[%3, %arg4] {index = 0 : i8, offset = 0 : si32} : memref<18x288xi8>, vector<32xi8>
      %11 = aievec.mac %1, %10, %8 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "0", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %12 = aievec.mac %1, %10, %9 {xoffsets = "0x00000000", xsquare = "0x1010", xstart = "32", xstep = "4", zoffsets = "0x43322110", zsquare = "0x2110", zstart = "4", zstep = "2"} : vector<64xi8>, vector<32xi8>, !aievec.acc<16xi48>
      %13 = aievec.srs %11 {shift = 10 : i8} : !aievec.acc<16xi48>, vector<16xi16>
      %14 = aievec.srs %12 {shift = 10 : i8} : !aievec.acc<16xi48>, vector<16xi16>
      %15 = aievec.concat %13, %14 : vector<16xi16>, vector<32xi16>
      %16 = aievec.select %15 {select = "0xcccccccc", xoffsets = "0x0c080400", xoffsets_hi = "0x0", xsquare = "0x1010", xstart = "0", yoffsets = "0x0c080400", yoffsets_hi = "0x0", ysquare = "0x1010", ystart = "4"} : vector<32xi16>, vector<32xi16>
      %17 = aievec.ext %16 {index = 0 : i8} : vector<32xi16>, vector<16xi16>
      %18 = aievec.pack %17 : vector<16xi16>, vector<16xi8>
      vector.transfer_write %18, %arg2[%arg3, %arg4] : vector<16xi8>, memref<16x256xi8>
    }
  }
  return
}

//CHECK-NEXT:  size_t v4 = 32;
//CHECK-NEXT:  size_t v5 = 0;
//CHECK-NEXT:  v64int8 v6;
//CHECK-NEXT:  int8_t * restrict r_v6_v2 = v2;
//CHECK-NEXT:  v6 = upd_w(v6, 0, *(v32int8 *)(r_v6_v2 + v5));
//CHECK-NEXT:  v6 = upd_w(v6, 1, *(v32int8 *)(r_v6_v2 + v4));
//CHECK-NEXT:  size_t v7 = 0;
//CHECK-NEXT:  size_t v8 = 16;
//CHECK-NEXT:  size_t v9 = 1;
//CHECK-NEXT:  for (size_t v10 = v7; v10 < v8; v10 += v9)
//CHECK-NEXT:  chess_prepare_for_pipelining
//CHECK-NEXT:  chess_loop_range(16, 16)
//CHECK-NEXT:  {
//CHECK-NEXT:    size_t v11 = 1;
//CHECK-NEXT:    size_t v12 = v10 + v11;
//CHECK-NEXT:    size_t v13 = 2;
//CHECK-NEXT:    size_t v14 = v10 + v13;
//CHECK-NEXT:    size_t v15 = 0;
//CHECK-NEXT:    size_t v16 = 256;
//CHECK-NEXT:    size_t v17 = 16;
//CHECK-NEXT:    for (size_t v18 = v15; v18 < v16; v18 += v17)
//CHECK-NEXT:    chess_prepare_for_pipelining
//CHECK-NEXT:    chess_loop_range(16, 16)
//CHECK-NEXT:    {
//CHECK-NEXT:      v32int8 v19 = *(v32int8 *)(v1 + 288*v10+v18);
//CHECK-NEXT:      v16acc48 v20 = mul16(v6, 0, 0x00000000, 4, 0x1010, v19, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v16acc48 v21 = mul16(v6, 0, 0x00000000, 4, 0x1010, v19, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v32int8 v22 = *(v32int8 *)(v1 + 288*v12+v18);
//CHECK-NEXT:      v20 = mac16(v20, v6, 16, 0x00000000, 4, 0x1010, v22, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v21 = mac16(v21, v6, 16, 0x00000000, 4, 0x1010, v22, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v32int8 v23 = *(v32int8 *)(v1 + 288*v14+v18);
//CHECK-NEXT:      v20 = mac16(v20, v6, 32, 0x00000000, 4, 0x1010, v23, 0, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v21 = mac16(v21, v6, 32, 0x00000000, 4, 0x1010, v23, 4, 0x43322110, 2, 0x2110);
//CHECK-NEXT:      v16int16 v24 = srs(v20, 10);
//CHECK-NEXT:      v16int16 v25 = srs(v21, 10);
//CHECK-NEXT:      v32int16 v26 = concat(v24, v25);
//CHECK-NEXT:      v32int16 v27 = select32(0xcccccccc, v26, 0, 0x0c080400, 0x0, 0x1010, 4, 0x0c080400, 0x0, 0x1010);
//CHECK-NEXT:      v16int16 v28 = ext_w(v27, 0);
//CHECK-NEXT:      v16int8 v29 = pack(v28);
//CHECK-NEXT:      *(v16int8 *)(v3 + 256*v10+v18) = v29;
//CHECK-NEXT:    }
//CHECK-NEXT:  }
