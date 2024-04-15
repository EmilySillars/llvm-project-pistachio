// https://github.com/openai/triton/pull/1866
// https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfforall-scfforallop
// https://mlir.llvm.org/docs/Tutorials/transform/Ch0/#tiling-and-loop-materialization

// mlir-opt TRUNK
// -test-linalg-transform-patterns=test-linalg-to-vector-patterns -empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize -bufferization-bufferize -tensor-bufferize -func-bufferize -finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref -convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts

//%0 = vector.reduction <add>, %0 : vector<8xf32> into f32

//mlir-opt scf-practice.mlir -convert-vector-to-llvm -split-input-file | FileCheck %s


func.func @main() {
     %v = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : vector<4xf32>
     %s = arith.constant 7.0 : f32
     %ind = arith.constant 2 : i32
     %0 = vector.insertelement %s, %v[%ind : i32] : vector<4xf32>
    
     vector.print %0 : vector<4xf32>

     %c0 = arith.constant 0 : index
     %c1 = arith.constant 1 : index
     %c8 = arith.constant 8 : index
     %init = arith.constant 0.0 : f32
     %zeroo = arith.constant 0 : i32
     //            for i = 0 to 8, i++, partial = 0
     %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%partial = %init) -> (f32) {
     %element = vector.extractelement %0[%i : index] : vector<4xf32>
     %updated = arith.addf %partial, %element : f32
     %dummy_v = arith.constant dense<[0.9999]> : vector<1xf32>
     %updated_as_v = vector.insertelement %updated, %dummy_v[%zeroo : i32] : vector<1xf32>
     vector.print %updated_as_v : vector<1xf32>
     scf.yield %updated : f32
     }
     
     %result_v = arith.constant dense<[0.2]> : vector<1xf32>
     %result_v_final = vector.insertelement %result, %result_v[%zeroo : i32] : vector<1xf32>

     vector.print %result_v : vector<1xf32>
     vector.print %0 : vector<4xf32>

  return
}

// func.func @simp(
//       %lhs : tensor<2xf32>
//     ) -> tensor<2xf32> {
//      %zero = arith.constant 0 : index
//      %one = arith.constant 1 : index
//      %c0f32 = arith.constant 77.0 : f32
//      %c0f33 = arith.constant 78.0 : f32
//      %empty =  tensor.empty() : tensor<2xf32>
//      %a = tensor.insert %c0f32 into %empty[%zero] : tensor<2xf32>
//      %b = tensor.insert %c0f33 into %a[%one] : tensor<2xf32>
//      %v = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : vector<4xf32>
//      %s = arith.constant 7.0 : f32
//      %ind = arith.constant 2 : i32
//      // pasted in vvvvvvvvvvvvvvvvvvv
//      %0 = vector.insertelement %s, %v[%ind : i32] : vector<4xf32>
    
//      vector.print %0 : vector<4xf32>

//      %c0 = arith.constant 0 : index
//      %c1 = arith.constant 1 : index
//      %c8 = arith.constant 8 : index
//      %init = arith.constant 0.0 : f32

//      %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%partial = %init) -> (f32) {
//      %element = vector.extractelement %0[%i : index] : vector<4xf32>
//      %updated = arith.addf %partial, %element : f32
//      scf.yield %updated : f32
//      }

//      %zeroo = arith.constant 0 : i32
//      %result_v = arith.constant dense<[0.2]> : vector<1xf32>
//      %result_v_final = vector.insertelement %result, %result_v[%zeroo : i32] : vector<1xf32>

//      vector.print %result_v : vector<1xf32>
//      vector.print %0 : vector<4xf32>
//      // pasted in ^^^^^^^^^^^^^^^6
//      return %b : tensor<2xf32>
// }
