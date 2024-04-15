// https://github.com/openai/triton/pull/1866
// https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfforall-scfforallop
// https://mlir.llvm.org/docs/Tutorials/transform/Ch0/#tiling-and-loop-materialization
// https://github.com/llvm/llvm-project/blob/1a4dd8d36206352220eb3306c3bdea79b6eeffc3/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir

module {
  memref.global "private" constant @__constant_1x2x3xf32 : memref<1x2x3xf32> = dense<[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [2.000000e+00, 3.000000e+00, 4.000000e+00]]]>
  func.func @main() {
    %cst = arith.constant 2.300000e+00 : f32
    %0 = memref.get_global @__constant_1x2x3xf32 : memref<1x2x3xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x5xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<1x4x5xf32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x4x5xf32>
    memref.copy %alloc, %alloc_0 : memref<1x4x5xf32> to memref<1x4x5xf32>
    %subview = memref.subview %alloc_0[0, 2, 0] [1, 2, 3] [1, 1, 1] : memref<1x4x5xf32> to memref<1x2x3xf32, strided<[20, 5, 1], offset: 10>>
    memref.copy %0, %subview : memref<1x2x3xf32> to memref<1x2x3xf32, strided<[20, 5, 1], offset: 10>>
    %cast = memref.cast %alloc_0 : memref<1x4x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<1x4x5xf32>
    memref.dealloc %alloc_0 : memref<1x4x5xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
}


// func.func @main() {
//   %const = arith.constant dense<[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]> : tensor<1x2x3xf32>
//   %dynamic = tensor.cast %const: tensor<1x2x3xf32> to tensor<1x?x3xf32>
//   %offset = arith.constant 2 : index
//   %cst = arith.constant 2.3 : f32
//   %c0 = arith.constant 0 : index
//   %out = tensor.pad %dynamic low[%c0, %offset, %c0] high[%c0, %c0, %offset]  {
//   ^bb0(%gen_arg1: index, %gen_arg2: index, %gen_arg3: index):
//     tensor.yield %cst : f32
//   } : tensor<1x?x3xf32> to tensor<1x?x?xf32>
//   %unranked = tensor.cast %out: tensor<1x?x?xf32> to tensor<*xf32>
//   call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
//   return
// }

// func.func private @printMemrefF32(%ptr : tensor<*xf32>)

// func.func @main() {
//      %v = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : vector<4xf32>
//      %s = arith.constant 7.0 : f32
//      %ind = arith.constant 2 : i32
//      %0 = vector.insertelement %s, %v[%ind : i32] : vector<4xf32>
    
//      vector.print %0 : vector<4xf32>

//      %c0 = arith.constant 0 : index
//      %c1 = arith.constant 1 : index
//      %c8 = arith.constant 8 : index
//      %init = arith.constant 0.0 : f32
//      %zeroo = arith.constant 0 : i32
//      //            for i = 0 to 8, i++, partial = 0
//      %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%partial = %init) -> (f32) {
//      %element = vector.extractelement %0[%i : index] : vector<4xf32>
//      %updated = arith.addf %partial, %element : f32
//      %dummy_v = arith.constant dense<[0.9999]> : vector<1xf32>
//      %updated_as_v = vector.insertelement %updated, %dummy_v[%zeroo : i32] : vector<1xf32>
//      vector.print %updated_as_v : vector<1xf32>
//      scf.yield %updated : f32
//      }
     
//      %result_v = arith.constant dense<[0.2]> : vector<1xf32>
//      %result_v_final = vector.insertelement %result, %result_v[%zeroo : i32] : vector<1xf32>

//      vector.print %result_v : vector<1xf32>
//      vector.print %0 : vector<4xf32>

//   return
// }
