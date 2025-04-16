#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

"builtin.module"() ({
  // MLIR function prototype for the external function defined in C
  "func.func"() <{function_type = (
  memref<2x2xi32, strided<[2,1], offset: ?>>) 
  -> (), sym_name = "print_memref_32_bit", sym_visibility = "private"}> ({}) {llvm.emit_c_interface}: () -> ()

  // MLIR function we plan to call from main
  "func.func"() <{function_type = (memref<2x2xi32, strided<[2,1], offset: ?>>,
                                   memref<2x2xi32, strided<[2,1], offset: ?>>,
                                   memref<2x2xi32, strided<[2,1], offset: ?>>) -> (), 
                  sym_name = "matmulAndPrint"}> ({
    ^bb0(%a: memref<2x2xi32, strided<[2,1], offset: ?>>,
         %b: memref<2x2xi32, strided<[2,1], offset: ?>>, 
         %c: memref<2x2xi32, strided<[2,1], offset: ?>>): 

    "linalg.matmul"(%a, %b, %c) ({
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
      %3 = "arith.muli"(%arg0, %arg1) {fastmath = #arith.fastmath<none>} : (i32, i32) -> i32
      %4 = "arith.addi"(%arg2, %3) {fastmath = #arith.fastmath<none>} : (i32, i32) -> i32
      "linalg.yield"(%4) : (i32) -> ()
    }) {linalg.memoized_indexing_maps = [#map, #map1, #map2], operand_segment_sizes = array<i32: 2, 1>} : (memref<2x2xi32, strided<[2,1], offset: ?>>, memref<2x2xi32, strided<[2,1], offset: ?>>, memref<2x2xi32, strided<[2,1], offset: ?>>) -> ()
    
    // print out the resulting matrix product,
    // using a function defined in C
    func.call @print_memref_32_bit(%c)
      :(memref<2x2xi32, strided<[2,1], offset: ?>>) -> ()
    
    "func.return"() : () -> ()
  }) {llvm.emit_c_interface}: () -> ()

}) : () -> ()