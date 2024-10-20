
"func.func"() ({
 ^bb0(%arg0: tensor<104x104xi8>, %arg1: tensor<104x104xi8>, %arg2: tensor<104x104xi32>):
  %0 = "linalg.generic"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: i8, %arg4: i8, %arg5: i32):
    %1 = "arith.extsi"(%arg3) : (i8) -> i32
    %2 = "arith.extsi"(%arg4) : (i8) -> i32
    %3 = "arith.muli"(%1, %2) : (i32, i32) -> i32
    %4 = "arith.addi"(%arg5, %3) : (i32, i32) -> i32
    "linalg.yield"(%4) : (i32) -> ()
  }) {indexing_maps = [
  affine_map<(d0, d1, d2) -> (d0, d2)>, 
  affine_map<(d0, d1, d2) -> (d2, d1)>, 
  affine_map<(d0, d1, d2) -> (d0, d1)>], 
  iterator_types = 
  [#linalg.iterator_type<parallel>, 
  #linalg.iterator_type<parallel>, 
  #linalg.iterator_type<reduction>], 
  operand_segment_sizes = array<i32: 2, 1>} 
  : (tensor<104x104xi8>, tensor<104x104xi8>, tensor<104x104xi32>) -> tensor<104x104xi32>
  "func.return"(%0) : (tensor<104x104xi32>) -> ()
}) {function_type = (tensor<104x104xi8>, tensor<104x104xi8>, tensor<104x104xi32>) -> tensor<104x104xi32>, sym_name = "matmul104x104"} : () -> ()

// %2 = linalg.generic {indexing_maps = [
//   affine_map<(d0, d1, d2) -> (d0, d2)>, 
//   affine_map<(d0, d1, d2) -> (d2, d1)>, 
//   affine_map<(d0, d1, d2) -> (d0, d1)>], 
//   iterator_types = ["parallel", "parallel", "reduction"]} 
//   ins(%extracted_slice, %extracted_slice_0 : tensor<26x8xi8>, tensor<8x8xi8>) 
//   outs(%extracted_slice_1 : tensor<26x8xi32>) {
// ^bb0(%in: i8, %in_2: i8, %out: i32):
//   %3 = arith.extsi %in : i8 to i32
//   %4 = arith.extsi %in_2 : i8 to i32
//   %5 = arith.muli %3, %4 : i32
//   %6 = arith.addi %out, %5 : i32
//   linalg.yield %6 : i32
// } -> tensor<26x8xi32>

[zigzag-tile] A generated op is: mlir-asm-printer: Verifying operation: func.func
%2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<26x8xi8>, tensor<8x8xi8>) outs(%extracted_slice_1 : tensor<26x8xi32>) {
^bb0(%in: i8, %in_2: i8, %out: i32):
  %3 = arith.extsi %in : i8 to i32
  %4 = arith.extsi %in_2 : i8 to i32
  %5 = arith.muli %3, %4 : i32
  %6 = arith.addi %out, %5 : i32
  linalg.yield %6 : i32
} -> tensor<26x8xi32>
[zigzag-tile] A generated op loop: mlir-asm-printer: Verifying operation: func.func
%1 = scf.forall (%arg3, %arg4, %arg5) = (0, 0, 0) to (104, 104, 104) step (26, 8, 8) shared_outs(%arg6 = %arg2) -> (tensor<104x104xi32>) {
  %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5] [26, 8] [1, 1] : tensor<104x104xi8> to tensor<26x8xi8>
  %extracted_slice_0 = tensor.extract_slice %arg1[%arg5, %arg4] [8, 8] [1, 1] : tensor<104x104xi8> to tensor<8x8xi8>
  %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg4] [26, 8] [1, 1] : tensor<104x104xi32> to tensor<26x8xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<26x8xi8>, tensor<8x8xi8>) outs(%extracted_slice_1 : tensor<26x8xi32>) {
  ^bb0(%in: i8, %in_2: i8, %out: i32):
    %3 = arith.extsi %in : i8 to i32
    %4 = arith.extsi %in_2 : i8 to i32
    %5 = arith.muli %3, %4 : i32
    %6 = arith.addi %out, %5 : i32
    linalg.yield %6 : i32
  } -> tensor<26x8xi32>
  scf.forall.in_parallel {
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::AtLeastNOperands<2>::Impl<Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OffsetSizeAndStrideOpInterface::Trait<Empty>)
    tensor.parallel_insert_slice %2 into %arg6[%arg3, %arg4] [26, 8] [1, 1] : tensor<26x8xi32> into tensor<104x104xi32>
  }
}