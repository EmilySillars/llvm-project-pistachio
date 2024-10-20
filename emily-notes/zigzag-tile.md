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
Helpful notes for Emily:
```
cd ../build-riscv; clear; ninja -j 20
```
```
cd ../emily-notes;clear;mlir-opt matmul104x104-as-generic-linalg.mlir -pass-pipeline='builtin.module(func.func(zigzag-tile{tiling-scheme=zigzag-tile-scheme.json}))' --debug --mlir-disable-threading | head -n -54
```
For reference:
```
/// Transformation information returned after tile and fuse.
struct SCFTileAndFuseResult {
  /// List of untiled operations that were fused with the tiled consumer.
  llvm::SetVector<Operation *> fusedProducers;
  /// List of tiled and fused operations generated. The first one in this list
  /// is guaranteed to be the tiled operations generated during tiling of the
  /// generated operation.
  llvm::SetVector<Operation *> tiledAndFusedOps;
  /// The `scf.for` operations that iterate over the tiles.
  SmallVector<LoopLikeOpInterface> loops;
  /// The replacement values to use for the tiled and fused operations.
  llvm::DenseMap<Value, Value> replacements;
};
```

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

Some notes:
```
     tilingOptions.setTileSizeComputationFunction(
          [&](OpBuilder &builder, auto &&...) {
            SmallVector<OpFoldResult> result;

            SmallVector<int64_t> l1Tiles(loweringConfig.getL1Tiles());
            for (int64_t value : l1Tiles)
              result.push_back(builder.getIndexAttr(value));

            size_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
            while (result.size() < numLoops)
              result.push_back(builder.getIndexAttr(0));

            return result;
          });
```

More notes:

```
static SmallVector<OpFoldResult>
threadTileSizeComputation(OpBuilder &builder, Operation *operation) {
  SmallVector<OpFoldResult> result;

  std::optional<IntegerAttr> attr = getConfigIntegerAttr(
      IREE::HAL::ExecutableTargetAttr::lookup(operation), "compute_cores");
  if (!attr)
    return result;

  auto computeOp = cast<TilingInterface>(operation);
  std::optional<unsigned> largestParallelDim;
  std::optional<int64_t> largestParallelSize;
  for (auto [iterType, range] :
       llvm::zip_equal(computeOp.getLoopIteratorTypes(),
                       computeOp.getIterationDomain(builder))) {
    // Not doing reduction tiling.
    if (iterType == utils::IteratorType::reduction) {
      result.push_back(builder.getIndexAttr(0));
      continue;
    }

    // Not tileable.
    if (getConstantIntValue(range.size) == 1) {
      result.push_back(builder.getIndexAttr(0));
      continue;
    }

    // Not tiling dynamic dimensions right now.
    std::optional<int64_t> size = getConstantIntValue(range.size);
    if (!size) {
      result.push_back(builder.getIndexAttr(0));
      continue;
    }

    if (!largestParallelSize || size > largestParallelSize) {
      largestParallelDim = result.size();
      largestParallelSize = size;
    }

    // Placeholder for later.
    result.push_back(builder.getIndexAttr(0));
  }

  if (largestParallelDim) {
    assert(largestParallelSize);
    result[*largestParallelDim] = builder.getIndexAttr(llvm::divideCeil(
        *largestParallelSize, attr->getValue().getSExtValue()));
  }
  return result;
}

```

Need to set our own tileSizeComputation function!

For now, assume that the linalg op has the same number of loops as the list of tilesizes?

Incorporate json parsing now?

Is there a way to do this super simply instead of parsing json for the moment, so we can at least see what we get out when calling the tiling func?

Yes. just set the tile sizes to a list of numbers defined as magic numbers.

## IV. What about memref.subviews? Or tensor slices?

## Old notes delete later

