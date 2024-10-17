# Linalg Tiling Pass

- [back to landing page](README.md)
- using [my dummy mlir-opt pass](https://github.com/EmilySillars/llvm-project-pistachio/tree/learn-llvm/EMILY-NOTES/add-dummy-pass#avocado-add-a-hello-world-pass-to-mlir-opt) as reference, as well as [my ad-hoc affine tiling pass](affine-ad-hoc-loop-tile.md)
- [also good reference](../mlir/test/lib/Dialect/Linalg/TestLinalgTransforms.cpp)
- [this one too](../mlir/lib/Dialect/Linalg/Transforms/DecomposeLinalgOps.cpp)
- [this one uses a pattern rewriter](../mlir/include/mlir/Dialect/Shape/Transforms/Passes.h); here is its [pass definition](../mlir/lib/Dialect/Shape/Transforms/RemoveShapeConstraints.cpp)

Example Run:

```
clear;mlir-opt matmul104x104-as-generic-linalg.mlir -pass-pipeline='builtin.module(func.func(zigzag-tile{tiling-scheme=zigzag-tile-scheme.json}))' --debug --mlir-disable-threading &> temp && cat temp | grep zigzag && rm temp
```

Running, but only to see debugging output:

```
clear;mlir-opt matmul104x104-as-generic-linalg.mlir -pass-pipeline='builtin.module(func.func(zigzag-tile{tiling-scheme=zigzag-tile-scheme.json}))' --debug --mlir-disable-threading | head -n -54
```



## I. Hoodle

## II. Motivating Example

**Desired Input:**

```
func.func @matmul104x104(%lhs: tensor<104x104xi8>, %rhs: tensor<104x104xi8>, %acc: tensor<104x104xi32>) -> tensor<104x104xi32> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<104x104xi8>, tensor<104x104xi8>)
    outs(%acc: tensor<104x104xi32>)
  -> tensor<104x104xi32>
  return %result: tensor<104x104xi32>
}
```

In generic MLIR syntax with `linalg.matmul` lowered to `linalg.generic`:

```
"func.func"() ({
 ^bb0(%arg0: tensor<104x104xi8>, %arg1: tensor<104x104xi8>, %arg2: tensor<104x104xi32>):
  %0 = "linalg.generic"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: i8, %arg4: i8, %arg5: i32):
    %1 = "arith.extsi"(%arg3) : (i8) -> i32
    %2 = "arith.extsi"(%arg4) : (i8) -> i32
    %3 = "arith.muli"(%1, %2) : (i32, i32) -> i32
    %4 = "arith.addi"(%arg5, %3) : (i32, i32) -> i32
    "linalg.yield"(%4) : (i32) -> ()
  }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operand_segment_sizes = array<i32: 2, 1>} : (tensor<104x104xi8>, tensor<104x104xi8>, tensor<104x104xi32>) -> tensor<104x104xi32>
  "func.return"(%0) : (tensor<104x104xi32>) -> ()
}) {function_type = (tensor<104x104xi8>, tensor<104x104xi8>, tensor<104x104xi32>) -> tensor<104x104xi32>, sym_name = "matmul104x104"} : () -> ()
```

+ also include desired zigzag tiling scheme as input:

  ```
  // recall:  O[a][b]+=I[a][c]*W[c][b]
  ===========================================================================================
  Temporal Loops                     I                  O                  W                  
  ===========================================================================================
  for c2 in [0, 4):                  l1                 l3                 l3                  C2 = 4
  -------------------------------------------------------------------------------------------
    for c1 in [0, 2):                l1                 l3                 l1                  C1 = 2
  -------------------------------------------------------------------------------------------
      for b1 in [0, 13):             l1                 l3                 l1                  B1 = 13
  -------------------------------------------------------------------------------------------
        for a1 in [0, 13):           l1                 l3                 rf_x1_thru_x31      A1 = 13
  ```

- which I manually encoded as the following json:
  ```
  {
      "bounds":[[13], [13], [4,2]],
      "order":[[2,0], [2,1], [1,0], [0,0]]
  }
  ```

**Desired Output:**

```

```

## III. Implementation

Next Steps:

- understand `SCFTilingOptions` defined in  [../mlir/include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h]()
- understand dominance info and the other steps inside `applyTileAndFuseToEachRoot` and `applyTileAndFuseToAll`
- what is a minimal `SCFTilingOptions` struct initialization I can fill out in order to tile my example matmul?

## IV. What about memref.subviews? Or tensor slices?

## Old notes delete later

