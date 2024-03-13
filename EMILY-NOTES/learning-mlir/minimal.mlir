// The 1D identity map, used below.
#map_1d_identity = affine_map<(m) -> (m)>

// Define a function @foo taking two tensor arguments `%lhs` and `%rhs` and returning a tensor.
func.func @foo(
      %lhs : tensor<10xf32>,
      %rhs : tensor<10xf32>
    ) -> tensor<10xf32> {

  // Return the function's first argument.
  return %lhs : tensor<10xf32>
}