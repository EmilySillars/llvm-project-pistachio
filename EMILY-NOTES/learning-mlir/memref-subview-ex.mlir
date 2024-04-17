// https://github.com/openai/triton/pull/1866
// https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfforall-scfforallop
// https://mlir.llvm.org/docs/Tutorials/transform/Ch0/#tiling-and-loop-materialization

// mlir-opt TRUNK
// -test-linalg-transform-patterns=test-linalg-to-vector-patterns -empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize -bufferization-bufferize -tensor-bufferize -func-bufferize -finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref -convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts

//%0 = vector.reduction <add>, %0 : vector<8xf32> into f32

//mlir-opt scf-practice.mlir -convert-vector-to-llvm -split-input-file | FileCheck %s

"builtin.module"() ({
  memref.global "private" constant @__constant_16x16f32 : memref<16x16xf32> = 
  dense<[
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.10, 10.0, 11.10, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.20, 10.0, 11.20, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.30, 10.0, 11.30, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.40, 10.0, 11.40, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.50, 10.0, 11.50, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.60, 10.0, 11.60, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.70, 10.0, 11.70, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.80, 10.0, 11.80, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.90, 10.0, 11.90, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.10, 10.0, 11.10, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.11, 10.0, 11.11, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.12, 10.0, 11.12, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.13, 10.0, 11.13, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.14, 10.0, 11.14, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.15, 10.0, 11.15, 12.0, 13.0, 14.0, 15.0, 16.0],
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.16, 10.0, 11.16, 12.0, 13.0, 14.0, 15.0, 16.0]]> 

func.func @main() {
    %const = memref.get_global @__constant_16x16f32 : memref<16x16xf32> 
   // %0 = memref.reinterpret_cast %alloc to offset: [0], sizes:[16,16], strides:[1,16] : memref<16x16xi8> to memref<16x16xi8, strided<[1, 16]>>
   // %0 = memref.cast  %const : memref<16x16xf32> to memref<16x16xf32, affine_map<(d0, d1) -> (d0 * 16 + d1)>>
// %0 = memref.alloc() : memref<8x16xf32, affine_map<(d0, d1) -> (d0 * 64 + d1 * 4 )>>
    %cast2 = memref.cast %const : memref<16x16xf32> to memref<*xf32>
    call @printMemrefF32(%cast2) : (memref<*xf32>) -> ()
    //#map0 = affine_map<(d0, d1)[s0] -> (d0 * 16 + d1+s0)>
    // %s2= memref.subview %const[8,8][8,8][1,1]  : memref<16x16xf32> to memref<8x8xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1+s0)>>
    %s2= memref.subview %const[8,8][8,8][1,1]  : memref<16x16xf32> to memref<8x8xf32, strided<[16, 1], offset: 136>>

    %cast = memref.cast %s2 : memref<8x8xf32, strided<[16, 1], offset: 136>> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
// expected result type to be 'memref<8x8xf32, strided<[16, 1], offset: 136>>' 
// or a rank-reduced version. (mismatch of result layout) mlir

// pppppppppppppppppppppppppppppppppppppppp
// Subview with constant offsets, sizes and strides.
// %1 = memref.subview %0[0, 2][4, 4][1, 1]
//   : memref<8x16xf32, affine_map<(d0, d1) -> (d0 * 64 + d1 * 4 )>> to
//     memref<4x4xf32, strided<[64, 4], offset: 8>>

// ppppppppppppppppppppppppppppppppppppppppppppp

    // set up the for loop
     %c0 = arith.constant 0 : index
     %c1 = arith.constant 2 : index
     %c8 = arith.constant 16 : index
     %init = arith.constant 0.0 : f32
     // setting up sub view variables
  //   %j = arith.constant 87 : index
   //  %k = arith.constant 87 : index
     %size0 = arith.constant 16 : index
     %size1 = arith.constant 2 : index
     %x = arith.constant 1 : index
     %y = arith.constant 1 : index
    // %z = arith.constant 87 : index
     //            for i = 0 to 8, i++, partial = 0, the accumulator
     %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%partial = %init) -> (f32) {
      // try to print subview of matrix

  //     %1 = memref.subview %0[0, %c1][%size0, %size1][%x, %y]
  // : memref<16x16xf32, affine_map<(d0, d1) -> (d0 * 16 + d1)>> to
  //   memref<16x2xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>>

  //     %1 = memref.subview %0[0, %c1][%size0, %size1][%x, %y]
  // : memref<16x16xf32, affine_map<(d0, d1) -> (d0 * 16 + d1)>> to
  //   memref<16x2xf32, strided<[1,1], offset: ?>>


    //memref<?x?xf32, strided<[?, ?], offset: ?>>
      // try to get a valid subiview
      // try to print it out!
        %one = arith.constant 1.0 : f32
        %updated = arith.addf %partial, %one : f32
        scf.yield %updated : f32
     }
     
     // put result of for loop inside a vector
     // so that we can print it out
     %zeroo = arith.constant 0 : index
     %dummy_v = arith.constant dense<[0.2]> : vector<1xf32>
     %result_v = vector.insertelement %result, %dummy_v[%zeroo : index] : vector<1xf32>
     vector.print %result_v : vector<1xf32>
  return
}
func.func private @printMemrefF32(memref<*xf32>)
}) : () -> ()

// %0 = memref.alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>

// // Subview with constant offsets, sizes and strides.
// %1 = memref.subview %0[0, 2, 0][4, 4, 4][1, 1, 1]
//   : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>> to
//     memref<4x4x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>>



// %0 = memref.alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>

// // Create a sub-view of "base" memref '%0' with dynamic offsets, sizes,
// // and strides.
// // Note that dynamic offsets are represented by the linearized dynamic
// // offset symbol 's0' in the subview memref layout map, and that the
// // dynamic strides operands, after being applied to the base memref
// // strides in each dimension, are represented in the view memref layout
// // map as symbols 's1', 's2' and 's3'.
// %1 = memref.subview %0[%i, %j, %k][%size0, %size1, %size2][%x, %y, %z]
//   : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>> to
//     memref<?x?x?xf32,
//       affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + d1 * s2 + d2 * s3 + s0)>>
