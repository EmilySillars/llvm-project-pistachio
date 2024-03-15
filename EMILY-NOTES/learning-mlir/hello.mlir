// The 1D identity map, used below.
#map_1d_identity = affine_map<(m) -> (m)>

// Running sh mlir-to-llvm.sh hello.mlir print-fake-tensors.c
// prints out the return value of this function
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

// TODO: print out result of this function!
func.func @simp2(
      %lhs : tensor<3x9xf32>
    ) -> tensor<3x9xf32> {
     %zero = arith.constant 0 : index
     %one = arith.constant 1 : index
     %c0f32 = arith.constant 77.0 : f32
     %c0f33 = arith.constant 78.0 : f32
     %empty =  tensor.empty() : tensor<3x9xf32>
     %a = tensor.insert %c0f32 into %empty[%zero, %zero] : tensor<3x9xf32>
     %b = tensor.insert %c0f33 into %a[%zero, %one] : tensor<3x9xf32>
  return %b : tensor<3x9xf32>
}

// TODO: print out result of this function!
func.func @simp3(
      %lhs : tensor<2x3x9xf32>
    ) -> tensor<2x3x9xf32> {
     %zero = arith.constant 0 : index
     %one = arith.constant 1 : index
     %c0f32 = arith.constant 77.0 : f32
     %c0f33 = arith.constant 78.0 : f32
     %empty =  tensor.empty() : tensor<2x3x9xf32>
     %a = tensor.insert %c0f32 into %empty[%zero, %zero, %zero] : tensor<2x3x9xf32>
     %b = tensor.insert %c0f33 into %a[%zero, %one, %zero] : tensor<2x3x9xf32>
  return %b : tensor<2x3x9xf32>
}

// TODO: print out result of this function!
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
  return %c : tensor<3xf32>
}
