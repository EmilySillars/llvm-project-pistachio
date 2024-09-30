#affine_map42 = affine_map<(d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)>

"builtin.module"() ({

func.func @main() {
  %N = arith.constant 5 : index
  %zero = arith.constant 0 : index
  %two = arith.constant 2 : index
   
  %nine = arith.constant 9 : i32
  %one = arith.constant 1 : i32
  // Use an affine mapping definition in an alloc operation, binding the
  // SSA value %N to the symbol s0.
  %a = memref.alloc()[%N] : memref<4x4xi32, #affine_map42>
  linalg.fill ins(%one : i32) outs(%a :memref<4x4xi32, #affine_map42>)
  memref.store %nine, %a[%zero,%two] : memref<4x4xi32, #affine_map42>
  //  %inputElt = memref.load %slice_I_L1[%i_a, %i_c] : memref<104x104xi8, strided<[104, 1], offset: ?>>
  //   //arg 3: set all to zero
  //   linalg.fill ins(%three : i8) outs(%alloc_strided : memref<16x16xi8, strided<[1, 16]>>)
  // memref.store %eightySeven, %alloc_strided[%thirteen,%thirteen] : memref<16x16xi8, strided<[1, 16]>>
  // %zero = arith.constant 0 : i32
  // %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32> 
  // linalg.fill ins(%zero : i32) outs(%alloc_0 :memref<16x16xi32>)

  //%const_a = arith.constant dense<[[1,2,3]]> : tensor<1x3xi32>
  // %casted = tensor.cast %const_a: tensor<1x3xi32> to tensor<*xi32>
  %elt = memref.load %a[%zero,%two]: memref<4x4xi32, #affine_map42>
  %elt_as_vec = memref.alloc(): memref<1xi32>
  linalg.fill ins(%elt : i32) outs(%elt_as_vec : memref<1xi32>)
  %casted2 = memref.cast %elt_as_vec: memref<1xi32> to memref<*xi32>
  call @printMemrefI32(%casted2) : (memref<*xi32>) -> ()
  // call @printMemrefI32(%casted) : (tensor<*xi32>) -> ()
  return
}

func.func private @printMemrefI32(memref<*xi32>)

// func.func private @printMemrefI32(tensor<*xi32>)

}) : () -> ()