// The 1D identity map, used below.
#map_1d_identity = affine_map<(m) -> (m)>

// Define a function @foo taking two tensor arguments `%lhs` and `%rhs` and returning a tensor.
func.func @foo(
      %lhs : tensor<3xf32>,
      %rhs : tensor<3xf32>
    ) -> tensor<3xf32> {
     %zero = arith.constant 0 : index
     %one = arith.constant 1 : index
     %two = arith.constant 2 : index
     %c0f32 = arith.constant 5.0 : f32
     %c0f33 = arith.constant 6.0 : f32
     %c0f34 = arith.constant 7.0 : f32
     %empty =  tensor.empty() : tensor<3xf32>
     %a = tensor.insert %c0f32 into %empty[%zero] : tensor<3xf32>
     %b = tensor.insert %c0f33 into %a[%one] : tensor<3xf32>
     %c = tensor.insert %c0f34 into %b[%two] : tensor<3xf32>
// %4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
  // Return the function's first argument.
  return %c : tensor<3xf32>
}