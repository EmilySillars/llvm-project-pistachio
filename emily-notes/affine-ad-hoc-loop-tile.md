# ad-hoc tiling pass for non-hyperrectangular loop nests

- [back to landing page](README.md)
- using [my dummy mlir-opt pass](https://github.com/EmilySillars/llvm-project-pistachio/tree/learn-llvm/EMILY-NOTES/add-dummy-pass#avocado-add-a-hello-world-pass-to-mlir-opt) as reference, as well as [LoopTiling.cpp](https://github.com/EmilySillars/llvm-project-pistachio/blob/tiling/mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp)

## I. What are non-hyperrectangular loop nests?

"loop nests whose bounds are affine functions of loop iteration variables (i.e., nonrectangular loop nests)" (Jiménez et al., 410)

## II. Motivating Example

**Desired Input:**

```
func.func @matmul104x104(
%arg0: memref<104x104xi8, strided<[?, ?], offset: ?>>, 
%arg1: memref<104x104xi8, strided<[?, ?], offset: ?>>, 
%arg2: memref<104x104xi32, strided<[?, ?], offset: ?>>) 
-> memref<104x104xi32, strided<[?, ?], offset: ?>> {
    affine.for %arg3 = 0 to 104 {
      affine.for %arg4 = 0 to 104 {
        affine.for %arg5 = 0 to 104 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<104x104xi8, strided<[?, ?], offset: ?>>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<104x104xi8, strided<[?, ?], offset: ?>>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<104x104xi32, strided<[?, ?], offset: ?>>
          %3 = arith.extsi %0 : i8 to i32
          %4 = arith.extsi %1 : i8 to i32
          %5 = arith.muli %3, %4 : i32
          %6 = arith.addi %2, %5 : i32
          affine.store %6, %arg2[%arg3, %arg4] : memref<104x104xi32, strided<[?, ?], offset: ?>>
        }
      }
    }
    return %arg2 : memref<104x104xi32, strided<[?, ?], offset: ?>>
 }
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
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 8)>
#map2 = affine_map<(d0) -> (d0 + 26)>
#map3 = affine_map<(d0) -> (d0 + 13)>

func.func @matmul104x104(
%arg0: memref<104x104xi8, strided<[?, ?], offset: ?>>, 
%arg1: memref<104x104xi8, strided<[?, ?], offset: ?>>, 
%arg2: memref<104x104xi32, strided<[?, ?], offset: ?>>) 
-> memref<104x104xi32, strided<[?, ?], offset: ?>> {
    affine.for %arg3 = 0 to 104 step 8 {
      affine.for %arg4 = 0 to 104 step 8 {
        affine.for %arg5 = 0 to 104 step 26 {
          affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
              affine.for %arg8 = #map(%arg5) to #map2(%arg5) step 13{ 
                affine.for %arg9 = #map(%arg8) to #map3(%arg8) {
                  %0 = affine.load %arg0[%arg6, %arg9] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                  %1 = affine.load %arg1[%arg9, %arg7] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                  %2 = affine.load %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
                  %3 = arith.extsi %0 : i8 to i32
                  %4 = arith.extsi %1 : i8 to i32
                  %5 = arith.muli %3, %4 : i32
                  %6 = arith.addi %2, %5 : i32
                  affine.store %6, %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
                } // end of hoodle for
              }
            }
          }
        }
      }
    }
    return %arg2 : memref<104x104xi32, strided<[?, ?], offset: ?>>
 }
```

## III. Implementation

Now I take in a file name, and read that file to get the tiling scheme!

```
clear;mlir-opt --affine-ad-hoc-loop-tile=tiling-scheme=zigzag-tile-scheme.json matmul104x104-as-affine.mlir --debug --mlir-disable-threading &> temp && cat temp | head -n -38 && rm temp
```



Best way to run when you need to see only your own print statements:

```
clear;mlir-opt --affine-ad-hoc-loop-tile=tiling-scheme=zigzag-tile-scheme.json matmul104x104-as-affine.mlir --debug --mlir-disable-threading &> temp && cat temp | grep affine-ad-hoc-loop-tile && rm temp
```



Problem:

```
In file included from /home/hoppip/llvm-project-pistachio/mlir/lib/Dialect/Affine/Transforms/AdHocLoopTiling.cpp:16:
/home/hoppip/llvm-project-pistachio/mlir/include/mlir/Dialect/Affine/AdHocLoopUtils.h:55:13: warning: unused function 'constructTiledLoopNest' [-Wunused-function]
   55 | static void constructTiledLoopNest(MutableArrayRef<AffineForOp> origLoops,
      |             ^~~~~~~~~~~~~~~~~~~~~~
/home/hoppip/llvm-project-pistachio/mlir/include/mlir/Dialect/Affine/AdHocLoopUtils.h:60:1: warning: unused function 'constructTiledIndexSetHyperRect' [-Wunused-function]
   60 | constructTiledIndexSetHyperRect(MutableArrayRef<AffineForOp> origLoops,
      | ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/hoppip/llvm-project-pistachio/mlir/include/mlir/Dialect/Affine/AdHocLoopUtils.h:65:13: warning: unused function 'moveLoopBodyImpl' [-Wunused-function]
   65 | static void moveLoopBodyImpl(AffineForOp src, AffineForOp dest,
      |             ^~~~~~~~~~~~~~~~
/home/hoppip/llvm-project-pistachio/mlir/include/mlir/Dialect/Affine/AdHocLoopUtils.h:70:13: warning: unused function 'moveLoopBody' [-Wunused-function]
   70 | static void moveLoopBody(AffineForOp src, AffineForOp dest);

```



````
clear;mlir-opt --affine-ad-hoc-loop-tile=tiling-scheme=zigzag-tile-scheme.json matmul104x104-as-affine.mlir --debug --mlir-disable-threading &> temp && cat temp | grep "\[affine-ad-hoc-loop-tile\]" && rm temp
````

Need to resolve following error:

```
[affine-ad-hoc-loop-tile] the options are  [ tiling-scheme=zigzag-tile-scheme.json ]
[affine-ad-hoc-loop-tile] the filename is  [ zigzag-tile-scheme.json ]
[affine-ad-hoc-loop-tile] file contains... [ {
    "bounds":[[13], [13], [4,2]],
    "order":[[2,0], [2,1], [1,0], [0,0]]
}
   ]
Expected<T> must be checked before access or destruction.
Unchecked Expected<T> contained error:
[1:0, byte=0]: Invalid UTF-8 sequencePLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
```



I need to parse the json file with something inside this file: 

```
#include "llvm/Support/JSON.h"
```



## Old notes delete later

Checking band size for `matmul104x104-as-affine.mlir` gives

```
Args: mlir-opt --affine-ad-hoc-loop-tile=tiling-scheme=zigzag-tile-scheme.json matmul104x104-as-affine.mlir --debug --mlir-disable-threading 
[affine-ad-hoc-loop-tile] the options are  [ tiling-scheme=zigzag-tile-scheme.json ]
[affine-ad-hoc-loop-tile] the filename is  [ zigzag-tile-scheme.json ]
[affine-ad-hoc-loop-tile] file contains... [ {
[affine-ad-hoc-loop-tile] the tile scheme we parsed from the json is...
[affine-ad-hoc-loop-tile] band size is  3 
```

checking band size for `matmul104x104-tiled-twice.mlir` gives

```
Args: mlir-opt --affine-ad-hoc-loop-tile=tiling-scheme=zigzag-tile-scheme.json matmul104x104-tiled-twice.mlir --debug --mlir-disable-threading 
[affine-ad-hoc-loop-tile] the options are  [ tiling-scheme=zigzag-tile-scheme.json ]
[affine-ad-hoc-loop-tile] the filename is  [ zigzag-tile-scheme.json ]
[affine-ad-hoc-loop-tile] file contains... [ {
[affine-ad-hoc-loop-tile] the tile scheme we parsed from the json is...
[affine-ad-hoc-loop-tile] band size is  6 

```





I need to take in a list of lists! How to do this on command line? Currently only list of uints works...

```
mlir-opt --affine-ad-hoc-loop-tile=tile-sizes=8,8,26,1,1,13 matmul104x104-as-affine.mlir 
```

```
   //AffineBound ub = origLoops[i].getUpperBound();

    // llvm::errs() << "Error: field labeled '" << listName
    //              << "' does not exist \n ";
    // exit(1);

    // OperandRange newLbOperands = origLoops[i].getLowerBoundOperands();
    // OperandRange newUbOperands = origLoops[i].getUpperBoundOperands();
    // newLoops[i].setLowerBound(newLbOperands, origLoops[i].getLowerBoundMap());
    // newLoops[i].setUpperBound(newUbOperands, origLoops[i].getUpperBoundMap());
    // // If the step size of original loop is x and tileSize is y then after
    // // tiling the tile space loops' step size becomes x*y.
    // newLoops[i].setStep(tileSizes[i] * origLoops[i].getStepAsInt());
    // AffineBound ub = origLoops[i].getUpperBound();
        // reference code snippets
    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    //     bool AffineForOp::hasConstantLowerBound() {
    //   return getLowerBoundMap().isSingleConstant();
    // }
    // bool AffineForOp::hasConstantUpperBound() {
    //   return getUpperBoundMap().isSingleConstant();
    // }
    // int64_t AffineForOp::getConstantLowerBound() {
    //   return getLowerBoundMap().getSingleConstantResult();
    // }
    // int64_t AffineForOp::getConstantUpperBound() {
    //   return getUpperBoundMap().getSingleConstantResult();
    // }
    // getTileSizes(band, &tileSizes); //SmallVectorImpl<unsigned> *tileSizes
    // forOp.getLowerBoundMap()
    //     auto *loopBody = forOp.getBody();
    // auto indVar = forOp.getInductionVar();
    // ValueRange iterArgs = forOp.getRegionIterArgs();
    // // This is the place where hoisted instructions would reside.
    // OpBuilder b(forOp.getOperation());
    // SmallPtrSet<Operation *, 8> opsToHoist;
    // SmallVector<Operation *, 8> opsToMove;
    // SmallPtrSet<Operation *, 8> opsWithUsers;
    // llvm::errs() << "Error: top-level value is not a JSON object: " << '\n';
    //   exit(1);
    // LLVM_DEBUG(llvm::dbgs()
    //            << "[" DEBUG_TYPE "] "
    //            << "the filename is  [ " << options.data() << " ]\n");
    // reference code snippets
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    // unsiged tileSize = original loop bound / bound
    // throw an error if bound is not divisible?
    // TODO: validation that the tiling scheme makes sense!
```

```
for (size_t i = 0; i < ts.order.size(); i++) {
    // LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Orig Loop is "<<
    // orig<<"\n");
    subloops.push_back(std::vector<AffineForOp>());
    for (size_t j = 0; j < ts.order[i].size(); j++) {
      // we assume order is really a list of "tuples"
      uint64_t uBound = ts.bounds[i][j];
      uint64_t tileSize;
      if (origLoops[i].hasConstantUpperBound()) {
        uint64_t oldUBound = origLoops[i].getConstantUpperBound();
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] oldUBound:[" << oldUBound << "] uBound:["<< uBound<<"] \n");
        tileSize = oldUBound / uBound;

      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE
                      "] no constant upper bound, so using default val of "
                   << 87 << "\n");
        tileSize = 87;
      }
      OpBuilder b(topLoop);
      AffineForOp pointLoop = b.create<AffineForOp>(loc, i, uBound);
       LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] step size proposed was  [" << tileSize << "] \n");
      pointLoop.setStep(tileSize);
      subloops.back().push_back(pointLoop);
      topLoop = pointLoop.getOperation();
      if (i == 0)
        innermostPointLoop = pointLoop;
    }
  }
```

something is very messed up here:

```
  size_t bad = 0;
  for (const auto &order : ts.order) {
    // we assume order is really a list of "2-elt-tuples"
    size_t loopIdx = order[0];
    size_t subloopIdx = order[1];
    subloops.push_back(std::vector<AffineForOp>());
    for (size_t j = 0; j < ts.order[i].size(); j++) {

      uint64_t uBound = ts.bounds[i][j];
      uint64_t tileSize;
      if (origLoops[i].hasConstantUpperBound()) {
        uint64_t oldUBound = origLoops[bad].getConstantUpperBound();
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] oldUBound:[" << oldUBound
                                << "] uBound:[" << uBound << "] \n");
        tileSize = oldUBound / uBound;

      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE
                      "] no constant upper bound, so using default val of "
                   << 87 << "\n");
        tileSize = 87;
      }
      OpBuilder b(topLoop);
      AffineForOp pointLoop = b.create<AffineForOp>(loc, i, uBound);
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] step size proposed was  ["
                              << tileSize << "] \n");
      pointLoop.setStep(tileSize);
      subloops.back().push_back(pointLoop);
      topLoop = pointLoop.getOperation();
    }
    bad++;
  }
```

