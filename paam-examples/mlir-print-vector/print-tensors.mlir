// Sources consulted:
// https://github.com/openai/triton/pull/1866
// https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfforall-scfforallop
// https://mlir.llvm.org/docs/Tutorials/transform/Ch0/#tiling-and-loop-materialization
// https://github.com/llvm/llvm-project/blob/1a4dd8d36206352220eb3306c3bdea79b6eeffc3/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir
// For more info, see documentation on tensor.pad:
// https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorpad-tensorpadop

func.func @main() {
  // create a 3D tensor with dimensions 1x2x3
  %const = arith.constant dense<[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]> : tensor<1x2x3xf32>
  // cast this tensor to a tensor with dynamic size in the second dimension
  %dynamic = tensor.cast %const: tensor<1x2x3xf32> to tensor<1x?x3xf32>
  %offset = arith.constant 2 : index // the number of elements to add as padding in a particular dimension
  %cst = arith.constant 2.3 : f32    // the value with which to pad the tensor
  %c0 = arith.constant 0 : index     // another number of elts to add as a padding in a particular dimension
  // pad tensor %dynamic and save the result in %out.
  // low[0, 2, 0] means "padd the front of the second dimension with two elements"
  // high[0, 0, 2] means "padd the back of the third dimension with two elements"
  %out = tensor.pad %dynamic low[%c0, %offset, %c0] high[%c0, %c0, %offset]  {
  ^bb0(%gen_arg1: index, %gen_arg2: index, %gen_arg3: index):
    tensor.yield %cst : f32
  } : tensor<1x?x3xf32> to tensor<1x?x?xf32>
  // cast the padded tensor to an unranked (dimensionless) tensor
  %unranked = tensor.cast %out: tensor<1x?x?xf32> to tensor<*xf32>
  // print out the tensor
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)