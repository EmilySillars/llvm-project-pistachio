
"builtin.module"() ({
  
// example function definition
func.func @matmul104x104(%lhs: tensor<104x104xi8>, %rhs: tensor<104x104xi8>, %acc: tensor<104x104xi32>) -> tensor<104x104xi32> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<104x104xi8>, tensor<104x104xi8>)
    outs(%acc: tensor<104x104xi32>)
  -> tensor<104x104xi32>
  return %result: tensor<104x104xi32>
}

// used by mlir-cpu-runner
func.func private @printMemrefF32(%ptr : tensor<*xf32>)

func.func @main() {
  %vec = arith.constant dense<[[2.0, 3.0]]> : tensor<1x2xf32>
  %unranked = tensor.cast %vec: tensor<1x2xf32> to tensor<*xf32>
  call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
  return
}

}) : () -> ()
