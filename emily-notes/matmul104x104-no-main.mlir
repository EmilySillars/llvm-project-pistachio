"builtin.module"() ({

func.func @matmul104x104(%lhs: tensor<104x104xi8>, %rhs: tensor<104x104xi8>, %acc: tensor<104x104xi32>) -> tensor<104x104xi32> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<104x104xi8>, tensor<104x104xi8>)
    outs(%acc: tensor<104x104xi32>)
  -> tensor<104x104xi32>
  return %result: tensor<104x104xi32>
}
}) : () -> ()