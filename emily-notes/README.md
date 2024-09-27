# What kinds of tiling does mlir-opt support?

0. make sure to set your environment variables correctly!
   ```
   export PATH=/home/hoppip/llvm-project-pistachio/build-riscv/bin:$PATH
   ```

   and for using `mlir-cpu-runner`, do

   ```
   export MLIR_CPU_RUNNER_LIBS=/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so
   ```

## Setup

1. ```
   cd llvm-project-pistachio
   ```

2. ```
   mkdir build-riscv
   ```

3. ```
   cd build-rsicv
   ```

4. ```
   cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU;RISCV" \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_USE_LINKER=lld \
   -DCMAKE_C_COMPILER=/usr/bin/clang \
   -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON
   ```

5. ```
   ninja -j 20
   ```

## Investigation

1. `mlir-opt --help | tile`:

   ```
         --affine-loop-tile                                     -   Tile affine loop nests
           --cache-size=<ulong>                                 - Set size of cache to tile for in KiB (default: 512)
           --separate                                           - Separate full and partial tiles (default: false)
           --tile-size=<uint>                                   - Use this tile size for all loops
           --tile-sizes=<uint>                                  - List of tile sizes for each perfect nest (overridden by -tile-size)
         --allocate-arm-sme-tiles                               -   Allocate SME tiles
           --parallel-loop-tile-sizes=<long>                    - Factors to tile parallel loops by
         --test-affine-parametric-tile                          -   Tile affine loops using SSA values as tile sizes
           --loop-type=<string>                                 - Specify the type of loops to generate: for, parallel or tiled_loop
           --peeled-loops=<long>                                - Loops to be peeled when test-tile-pattern
           --tile-sizes=<long>                                  - Linalg tile sizes for test-tile-pattern
   
   ```

`test-tile-pattern` is mentioned as part of test-linalg [../mlir/test/lib/Dialect/Linalg/TestLinalgTransforms.cpp](../mlir/test/lib/Dialect/Linalg/TestLinalgTransforms.cpp)

2. `mlir-opt --help | grep tiling`:

   ```
         --scf-parallel-loop-tiling                             -   Tile parallel loops
           --no-min-max-bounds                                  - Perform tiling with fixed upper bound with inbound check inside the internal loops
         --test-extract-fixed-outer-loops                       -   test application of parametric tiling to the outer loops so that the ranges of outer loops become static
   ```

3. `mlir-opt --help | grep linalg`:

   ```
         --convert-elementwise-to-linalg                        -   Convert ElementwiseMappable ops to linalg
         --convert-linalg-to-affine-loops                       -   Lower the operations from the linalg dialect into affine loops
         --convert-linalg-to-loops                              -   Lower the operations from the linalg dialect into loops
         --convert-linalg-to-parallel-loops                     -   Lower the operations from the linalg dialect into parallel loops
         --convert-linalg-to-std                                -   Convert the operations from the linalg dialect into the Standard dialect
         --convert-tensor-to-linalg                             -   Convert some Tensor dialect ops to Linalg dialect
         --linalg-bufferize                                     -   Bufferize the linalg dialect
         --linalg-detensorize                                   -   Detensorize linalg ops
         --linalg-fold-unit-extent-dims                         -   Remove unit-extent dimension in Linalg ops on tensors
         --linalg-fuse-elementwise-ops                          -   Fuse elementwise operations on tensors
         --linalg-generalize-named-ops                          -   Convert named ops into generic ops
         --linalg-inline-scalar-operands                        -   Inline scalar operands into linalg generic ops
         --linalg-named-op-conversion                           -   Convert from one named linalg op to another.
       =only-generic                                      -   Run only on linalg.generic operations.
       =except-generic                                    -   Run on operations expect linalg.generic (e.g., foreach)
         --test-linalg-data-layout-propagation                  -   Test data layout propagation
         --test-linalg-decompose-ops                            -   Test Linalg decomposition patterns
         --test-linalg-drop-unit-dims                           -   
         --test-linalg-elementwise-fusion-patterns              -   Test Linalg element wise operation fusion patterns
           --fuse-with-reshape-by-collapsing                    - Test linalg expand_shape -> generic fusion patterns that collapse the iteration space of the consumer
           --fuse-with-reshape-by-collapsing-control            - Test controlling the linalg expand_shape -> generic fusion patterns that collapse the iteration space of the consumer
         --test-linalg-greedy-fusion                            -   Test Linalg fusion by applying a greedy test transformation.
         --test-linalg-pad-fusion                               -   Test PadOp fusion
         --test-linalg-transform-patterns                       -   Test Linalg transformation patterns by applying them greedily.
           --test-bubble-up-extract-slice-op-pattern            - Test rewrite of linalgOp + extract_slice into extract_slice + linalgOp
           --test-linalg-to-vector-patterns                     - Test a set of patterns that rewrite a linalg contraction in vector.contract form
           --test-swap-extract-slice-with-fill-pattern          - Test patterns to swap tensor.extract_slice(linalg.fill())
         --tosa-to-linalg                                       -   Lower TOSA to LinAlg on tensors
         --tosa-to-linalg-named                                 -   Lower TOSA to LinAlg named operations
           --prefer-conv2d-kernel-layout-hwcf                   - Prefer generating linalg.conv_2d_nhwc_hwcf over linalg.conv_2d_nhwc_fhwc
         --test-lower-to-llvm                                   -   An example of pipeline to lower the main dialects (arith, linalg, memref, scf, vector) down to LLVM.
         --tosa-to-linalg-pipeline                              -   The default pipeline for converting TOSA operators to the equivalent operations using the tensor operations in LinAlg as well as LinAlg named operations.
   ```

### linalg-tiling?

Files of interest:

- [llvm-project-pistachio/mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h](../mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h)
- [llvm-project-pistachio/mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp](../mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp)
- [llvm-project-pistachio/mlir/test/lib/Dialect/Linalg/TestLinalgTransforms.cpp](../mlir/test/lib/Dialect/Linalg/TestLinalgTransforms.cpp)

```
mlir-opt --linalg-tiling matmul104x104.mlir
```

### Affine tiling?

### Scf tiling?

## Examples

