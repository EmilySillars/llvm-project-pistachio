// The 1D identity map, used below.
#map_1d_identity = affine_map<(m) -> (m)>

func.func @simp(
      %lhs : tensor<2xf32>
    ) -> tensor<2xf32> {
     %zero = arith.constant 0 : index
     %one = arith.constant 1 : index
     %c0f32 = arith.constant 77.0 : f32
     %c0f33 = arith.constant 78.0 : f32
     %empty =  tensor.empty() : tensor<2xf32>
     %a = tensor.insert %c0f32 into %empty[%zero] : tensor<2xf32>
     %b = tensor.insert %c0f33 into %a[%one] : tensor<2xf32>
  return %b : tensor<2xf32>
}

func.func @simp2(
      %lhs : tensor<9xf32>
    ) -> tensor<9xf32> {
     %zero = arith.constant 0 : index
     %one = arith.constant 1 : index
     %c0f32 = arith.constant 77.0 : f32
     %c0f33 = arith.constant 78.0 : f32
     %empty =  tensor.empty() : tensor<9xf32>
     %a = tensor.insert %c0f32 into %empty[%zero] : tensor<9xf32>
     %b = tensor.insert %c0f33 into %a[%one] : tensor<9xf32>
  return %b : tensor<9xf32>
}

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

// func.func @foo2(
//       %lhs : tensor<7xf32>,
//       %rhs : tensor<9xf32>
//     ) -> tensor<3xf32> {
//      %zero = arith.constant 0 : index
//      %one = arith.constant 1 : index
//      %two = arith.constant 2 : index
//      %c0f32 = arith.constant 5.0 : f32
//      %c0f33 = arith.constant 6.0 : f32
//      %c0f34 = arith.constant 7.0 : f32
//      %empty =  tensor.empty() : tensor<3xf32>
//      %a = tensor.insert %c0f32 into %empty[%zero] : tensor<3xf32>
//      %b = tensor.insert %c0f33 into %a[%one] : tensor<3xf32>
//      %c = tensor.insert %c0f34 into %b[%two] : tensor<3xf32>
// // %4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
//   // Return the function's first argument.
//   return %c : tensor<3xf32>
// }