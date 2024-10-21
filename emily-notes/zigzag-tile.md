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

Note that this means the tile sizes we want are

```
a1_tile_sz = 8
b1_tile_sz = 8
c2_tile_sz = 26
c1_tile_sz = 13
```

which for now we simplify to

`8,8,26` and don't worry about second level tiling for the moment.

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

Seems like the easiest thing is to tile level by level, replacing ops in function each time, and then starting anew (instead of trying to pull out a partially tiled linalg generic body from a loop)

## IV. What kind of tiling do I get out?

original:

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
```

with set of tile sizes `tiles sizes = {8}, {8}, {26}}`:

```
%1 = scf.forall (%arg3, %arg4, %arg5) = (0, 0, 0) to (104, 104, 104) step (8, 8, 26) shared_outs(%arg6 = %arg2) -> (tensor<104x104xi32>) {
  %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5] [8, 26] [1, 1] : tensor<104x104xi8> to tensor<8x26xi8>
  %extracted_slice_0 = tensor.extract_slice %arg1[%arg5, %arg4] [26, 8] [1, 1] : tensor<104x104xi8> to tensor<26x8xi8>
  %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg4] [8, 8] [1, 1] : tensor<104x104xi32> to tensor<8x8xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<8x26xi8>, tensor<26x8xi8>) outs(%extracted_slice_1 : tensor<8x8xi32>) {
  ^bb0(%in: i8, %in_2: i8, %out: i32):
    %3 = arith.extsi %in : i8 to i32
    %4 = arith.extsi %in_2 : i8 to i32
    %5 = arith.muli %3, %4 : i32
    %6 = arith.addi %out, %5 : i32
    linalg.yield %6 : i32
  } -> tensor<8x8xi32>
  scf.forall.in_parallel {
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OpTrait::AtLeastNOperands<2>::Impl<Empty>)
ImplicitTypeIDRegistry::lookupOrInsert(mlir::OffsetSizeAndStrideOpInterface::Trait<Empty>)
    tensor.parallel_insert_slice %2 into %arg6[%arg3, %arg4] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<104x104xi32>
  }
}
```

followed by set of tile sizes `tiles sizes = {0}, {0}, {13}}`:

```
%1 = scf.forall (%arg3) = (0) to (104) step (13) shared_outs(%arg4 = %arg2) -> (tensor<104x104xi32>) {
  %extracted_slice = tensor.extract_slice %arg0[0, %arg3] [104, 13] [1, 1] : tensor<104x104xi8> to tensor<104x13xi8>
  %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0] [13, 104] [1, 1] : tensor<104x104xi8> to tensor<13x104xi8>
  %extracted_slice_1 = tensor.extract_slice %arg4[0, 0] [104, 104] [1, 1] : tensor<104x104xi32> to tensor<104x104xi32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<104x13xi8>, tensor<13x104xi8>) outs(%extracted_slice_1 : tensor<104x104xi32>) {
  ^bb0(%in: i8, %in_2: i8, %out: i32):
    %4 = arith.extsi %in : i8 to i32
    %5 = arith.extsi %in_2 : i8 to i32
    %6 = arith.muli %4, %5 : i32
    %7 = arith.addi %out, %6 : i32
    linalg.yield %7 : i32
  } -> tensor<104x104xi32>
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %3 into %arg4[0, 0] [104, 104] [1, 1] : tensor<104x104xi32> into tensor<104x104xi32>
  }
}
```

what about instead following with tile sizes `tiles sizes = {1}, {1}, {13` ?

```
%1 = scf.forall (%arg3, %arg4, %arg5) = (0, 0, 0) to (104, 104, 104) step (1, 1, 13) shared_outs(%arg6 = %arg2) -> (tensor<104x104xi32>) {
  %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5] [1, 13] [1, 1] : tensor<104x104xi8> to tensor<1x13xi8>
  %extracted_slice_0 = tensor.extract_slice %arg1[%arg5, %arg4] [13, 1] [1, 1] : tensor<104x104xi8> to tensor<13x1xi8>
  %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg4] [1, 1] [1, 1] : tensor<104x104xi32> to tensor<1x1xi32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<1x13xi8>, tensor<13x1xi8>) outs(%extracted_slice_1 : tensor<1x1xi32>) {
  ^bb0(%in: i8, %in_2: i8, %out: i32):
    %4 = arith.extsi %in : i8 to i32
    %5 = arith.extsi %in_2 : i8 to i32
    %6 = arith.muli %4, %5 : i32
    %7 = arith.addi %out, %6 : i32
    linalg.yield %7 : i32
  } -> tensor<1x1xi32>
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %3 into %arg6[%arg3, %arg4] [1, 1] [1, 1] : tensor<1x1xi32> into tensor<104x104xi32>
  }
}

```





## Old notes delete later

```
// recall:  O[a][b]+=I[a][c]*W[c][b]
===========================================================================================
Temporal Loops                     I                  O                  W                  
===========================================================================================
for c2 in [0, 4):                  l1                 l3                 l3                  C2 = 4, tile_sz = 26
-------------------------------------------------------------------------------------------
O <104x104xi32>
I <104x26xi8>
W <26x104xi8>
  for c1 in [0, 2):                l1                 l3                 l1                  C1 = 2, tile_sz = 13
-------------------------------------------------------------------------------------------
O <104x104xi32>
I <104x13xi8>
W <13x104xi8>

    for b1 in [0, 13):             l1                 l3                 l1                  B1 = 13, tile_sz = 8
-------------------------------------------------------------------------------------------
O <104x8xi32>
I <104x13xi8>
W <13x8xi8>
      for a1 in [0, 13):           l1                 l3                 rf_x1_thru_x31      A1 = 13, tile_sz = 8
-------------------------------------------------------------------------------------------
O <8x8xi32>
I <8x13xi8>
W <13x8xi8>
        for b0 in [0, 8):          rf_x1_thru_x31     l1                 rf_x1_thru_x31      B0 = 8, tile_sz = 1
-------------------------------------------------------------------------------------------
          for c0 in [0, 13):       rf_x1_thru_x31     rf_x1_thru_x31     rf_x1_thru_x31      C0 = 13, tile_sz = 1
-------------------------------------------------------------------------------------------
===========================================================================================
Spatial Loops                                                                              
===========================================================================================
            parfor a0 in [0, 8):                                                             A0 = 8    
-------------------------------------------------------------------------------------------
```

Or if the second level tiling of C was never done, we would have:

```
O <8x8xi32>
I <8x26xi8>
W <26x8xi8>
```

Thoughts:

The linalg tiling function sort of tiles how I expect. It does tile according to the size where each tile size passed in refers to a dimension to tile, but the linalg tiling function only generates tensor slices for the smallest tile size, which is NOT  ENOUGH when you want to copy a larger size tile from L3 to L1, right?

Can I get second level tiling by passing in MORE tile sizes? For example, pass in something like `8,8,26,0,0,13`? Maybe instead of passing in more tile sizes, I can tile twice?

```
8,8,26
```

followed by

```
0,0,13
```

?
