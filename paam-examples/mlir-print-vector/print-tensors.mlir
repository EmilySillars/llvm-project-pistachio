// Sources consulted:
// https://github.com/openai/triton/pull/1866
// https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfforall-scfforallop
// https://mlir.llvm.org/docs/Tutorials/transform/Ch0/#tiling-and-loop-materialization
// https://github.com/llvm/llvm-project/blob/1a4dd8d36206352220eb3306c3bdea79b6eeffc3/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir

func.func @main() {
  %const = arith.constant dense<[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]> : tensor<1x2x3xf32>
  %dynamic = tensor.cast %const: tensor<1x2x3xf32> to tensor<1x?x3xf32>
  %offset = arith.constant 2 : index
  %cst = arith.constant 2.3 : f32
  %c0 = arith.constant 0 : index
  %out = tensor.pad %dynamic low[%c0, %offset, %c0] high[%c0, %c0, %offset]  {
  ^bb0(%gen_arg1: index, %gen_arg2: index, %gen_arg3: index):
    tensor.yield %cst : f32
  } : tensor<1x?x3xf32> to tensor<1x?x?xf32>
  %unranked = tensor.cast %out: tensor<1x?x?xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)