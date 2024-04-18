// https://github.com/openai/triton/pull/1866
// https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfforall-scfforallop
// https://mlir.llvm.org/docs/Tutorials/transform/Ch0/#tiling-and-loop-materialization
// https://github.com/llvm/llvm-project/blob/1a4dd8d36206352220eb3306c3bdea79b6eeffc3/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir

// https://discourse.llvm.org/t/reasoning-about-memref-mutability/3830

//clear;sh run-func-memrefs.sh print-memrefs-qmat.mlir main

"builtin.module"() ({
  "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "simple_matmul"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1,16]>>, %arg2: memref<16x16xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
      %1 = "arith.extsi"(%arg3) : (i8) -> i32
      %2 = "arith.subi"(%1, %arg5) : (i32, i32) -> i32
      %3 = "arith.extsi"(%arg4) : (i8) -> i32
      %4 = "arith.subi"(%3, %arg6) : (i32, i32) -> i32
      %5 = "arith.muli"(%2, %4) : (i32, i32) -> i32
      %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32
      "linalg.yield"(%6) : (i32) -> ()
    }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1,16]>>, i32, i32, memref<16x16xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()

  // %1 = memref.transpose %0 (i, j) -> (j, i) : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>>

  "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32, strided<[16,1]>>) -> (), sym_name = "simple_matmul_tiled"}> ({
  ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1,16]>>, %arg2: memref<16x16xi32, strided<[16,1]>>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %zero = arith.constant 0 : index
    %one = arith.constant 1: index
    %sixteen = arith.constant 16 : index
    %two = arith.constant 2 : index
    %alloc = memref.alloc() {alignment = 1 : i64} : memref<2x2xi32> // fake result value
    linalg.fill ins(%0 : i32) outs(%alloc : memref<2x2xi32>)
    %arg1_t = memref.transpose %arg1 (i, j) -> (j, i): memref<16x16xi8, strided<[1,16]>> to  memref<16x16xi8, strided<[16,1]>>
    %fake_arg2 = memref.alloc() {alignment = 1 : i64} : memref<16x16xi32> // fake result value
    //%i_blk_sz = arith.constant 2 : index
    // enter FOR LOOP
    scf.for %k = %zero to %sixteen step %two iter_args() -> () {
    scf.for %j = %zero to %one step %one iter_args() -> () {
    
    scf.for %i = %zero to %sixteen step %two iter_args() -> () {
    // pull out left tile
    %leftTile = memref.subview %arg0[%i,%j][2,16][1,1] : memref<16x16xi8> to memref<2x16xi8, strided<[16, 1], offset: ?>>
    %leftTileCasted = memref.cast %leftTile : memref<2x16xi8, strided<[16, 1], offset: ?>> to memref<2x16xi8>
    // pull out right tile
    %rightTile = memref.subview %arg1_t[%j,%k][16,2][1,1] : memref<16x16xi8, strided<[16,1]>> to memref<16x2xi8, strided<[16,1], offset: ?>>
    %rightTileCasted = memref.cast %rightTile : memref<16x2xi8, strided<[16,1], offset: ?>> to memref<16x2xi8, strided<[16,1]>>
    // pull out output tile
    %outputTile = memref.subview %arg2[%i,%k][2,2][1,1] : memref<16x16xi32, strided<[16,1]>> to memref<2x2xi32, strided<[16,1], offset: ?>>
    %outputTileCasted = memref.cast %outputTile : memref<2x2xi32, strided<[16,1], offset: ?>> to memref<2x2xi32, strided<[16,1]>>
  
    //feed computation to linalg generic (accelerator workload)
    "linalg.generic"(%leftTileCasted, %rightTileCasted, %0, %0, %outputTileCasted) <{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>, 
        affine_map<(d0, d1, d2) -> (d2, d1)>, 
        affine_map<(d0, d1, d2) -> ()>, 
        affine_map<(d0, d1, d2) -> ()>, 
        affine_map<(d0, d1, d2) -> (d0, d1)>], 
        iterator_types = [
          #linalg.iterator_type<parallel>, 
          #linalg.iterator_type<parallel>, 
          #linalg.iterator_type<reduction>], 
          operandSegmentSizes = array<i32: 4, 1>}> ({
    ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
      %1 = "arith.extsi"(%arg3) : (i8) -> i32
      %2 = "arith.subi"(%1, %arg5) : (i32, i32) -> i32
      %3 = "arith.extsi"(%arg4) : (i8) -> i32
      %4 = "arith.subi"(%3, %arg6) : (i32, i32) -> i32
      %5 = "arith.muli"(%2, %4) : (i32, i32) -> i32
      %6 = "arith.addi"(%arg7, %5) : (i32, i32) -> i32
      func.call @myPrintI32(%6) : (i32 )-> ()
      "linalg.yield"(%6) : (i32) -> ()
    }) : (memref<2x16xi8>, memref<16x2xi8, strided<[16,1]>>, i32, i32, memref<2x2xi32, strided<[16,1]>>) -> ()
    %alloc3 = memref.cast %alloc :  memref<2x2xi32> to memref<*xi32>
    func.call @printMemrefI32(%alloc3) : (memref<*xi32>) -> ()
    } // end of i for
    } // end of j for
    } // end of k for

    %alloc2 = memref.cast %alloc :  memref<2x2xi32> to memref<*xi32>
    func.call @printMemrefI32(%alloc2) : (memref<*xi32>) -> ()
    memref.dealloc %alloc : memref<2x2xi32>
    memref.dealloc %fake_arg2 : memref<16x16xi32>
    "func.return"() : () -> ()
  }) : () -> ()

memref.global "private" constant @__constant_16x16f32 : memref<16x16xi8> = 
dense<[
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]> 

func.func @main() {
  // arg 1: set to global constant
  %0 = memref.get_global @__constant_16x16f32 : memref<16x16xi8>

  //arg 2: set all to three
  %three = arith.constant 3 : i8
  %alloc = memref.alloc() {alignment = 1 : i64} : memref<16x16xi8>
  %alloc_strided = memref.reinterpret_cast %alloc to offset: [0], sizes:[16,16], strides:[1,16] : memref<16x16xi8> to memref<16x16xi8, strided<[1, 16]>>
  linalg.fill ins(%three : i8) outs(%alloc_strided : memref<16x16xi8, strided<[1, 16]>>)

  //arg 3: set all to zero
  %zero = arith.constant 0 : i32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32> 
  linalg.fill ins(%zero : i32) outs(%alloc_0 :memref<16x16xi32>)

  // another arg3: set all to zero
  %alloc_00 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32> 
  linalg.fill ins(%zero : i32) outs(%alloc_00 :memref<16x16xi32>)
  %alloc_000 = memref.reinterpret_cast %alloc_00 to offset: [0], sizes:[16,16], strides:[16,1]: memref<16x16xi32> to  memref<16x16xi32, strided<[16,1]>>

  //call matmul
  call @simple_matmul(%0, %alloc_strided, %alloc_0) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> ()

  //call matmul
  call @simple_matmul_tiled(%0, %alloc_strided, %alloc_000) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32,strided<[16,1]>>) -> ()

  // print result of simple_matmul
  %cast2 = memref.cast %alloc_0 : memref<16x16xi32> to memref<*xi32>
  call @printMemrefI32(%cast2) : (memref<*xi32>) -> ()

  // print result of simple_matmul_tiled
  %cast3 = memref.cast %alloc_00 : memref<16x16xi32> to memref<*xi32>
  call @printMemrefI32(%cast3) : (memref<*xi32>) -> ()

  //clean up
  memref.dealloc %alloc : memref<16x16xi8>
  memref.dealloc %alloc_0 : memref<16x16xi32>
  return
}
func.func private @printMemrefI32(memref<*xi32>)
// helper to print out an f32
func.func@myPrintF32(%arg0: f32){
     %zeroo = arith.constant 0 : index
     %dummy_v = arith.constant dense<[0.2]> : vector<1xf32>
     %result_v = vector.insertelement %arg0, %dummy_v[%zeroo : index] : vector<1xf32>
     vector.print %result_v : vector<1xf32>
     return
}

// helper to print out an i32
func.func@myPrintI32(%arg0: i32){
     %zeroo = arith.constant 0 : index
     %dummy_v = arith.constant dense<[0]> : vector<1xi32>
     %result_v = vector.insertelement %arg0, %dummy_v[%zeroo : index] : vector<1xi32>
     vector.print %result_v : vector<1xi32>
     return
}

// helper to print out an index
func.func@myPrintIndex(%arg0: index){
     %zeroo = arith.constant 0 : index
     %dummy_v = arith.constant dense<[0]>: vector<1xindex>
     %result_v = vector.insertelement %arg0, %dummy_v[%zeroo : index] : vector<1xindex>
     vector.print %result_v : vector<1xindex>
     return
}
}) : () -> ()