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
    // set up the for loop
     %zero = arith.constant 0 : index
     %one = arith.constant 1 : index
     %sixteen = arith.constant 16 : index
     %two = arith.constant 2 : index

     scf.for %i = %zero to %sixteen step %two iter_args() -> () {
        // extract subview (tile)
        %tile = memref.subview %const[%zero,%i][16,2][1,1]  : memref<16x16xf32> to memref<16x2xf32, strided<[16, 1], offset: ?>>
        // print the tile
        %cast3 = memref.cast %tile : memref<16x2xf32, strided<[16, 1], offset: ?>> to memref<*xf32>
        func.call @printMemrefF32(%cast3) : (memref<*xf32>) -> ()
        //func.call @myPrintIndex(%i) : (index) -> ()
     }
    return
}

func.func private @printMemrefF32(memref<*xf32>)

// helper to print out an f32
func.func@myPrintF32(%arg0: f32){
     %zeroo = arith.constant 0 : index
     %dummy_v = arith.constant dense<[0.2]> : vector<1xf32>
     %result_v = vector.insertelement %arg0, %dummy_v[%zeroo : index] : vector<1xf32>
     vector.print %result_v : vector<1xf32>
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
