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

4. `mlir-opt --help | grep transform`

   ```
         --affine-loop-normalize                                 -   Apply normalization transformations to affine loop-like ops
         --test-create-vector-broadcast                          -   Test optimization transformations for transfer ops
         --test-linalg-greedy-fusion                             -   Test Linalg fusion by applying a greedy test transformation.
         --test-linalg-transform-patterns                        -   Test Linalg transformation patterns by applying them greedily.
           --test-generalize-pad-tensor                          - Test transform pad tensor by copying with generic ops
           --test-generalize-tensor-pack                         - Test transform that generalizes pack ops into a sequence of tensor and Linalg ops
           --test-generalize-tensor-unpack                       - Test transform that generalizes unpack ops into a sequence of tensor and Linalg ops
         --test-loop-unrolling                                   -   Tests loop unrolling transformation
         --test-multi-buffering                                  -   Test multi buffering transformation
         --test-scf-parallel-loop-collapsing                     -   Test parallel loops collapsing transformation
           --annotate                                            - Annote operations during loop pipelining transformation
         --test-tensor-transform-patterns                        -   Test Tensor transformation patterns by applying them greedily.
           --test-tracking-listener                              - Test tensor TrackingListener for the transform dialect
           --tile-consumer-and-fuse-producer-using-scf-for       - Test tile and fuse transformation using TilingInterface with scf.for operations
           --tile-consumer-fuse-and-yield-producer-using-scf-for - Test tile and fuse transformation while yielding fused producer replacements using TilingInterface with scf.for operations
         --test-transform-dialect-erase-schedule                 -   erase transform dialect schedule from the IR
         --test-transform-dialect-interpreter                    -   apply transform dialect operations one by one
           --debug-payload-root-tag=<string>                     - Select the operation with 'transform.target_tag' attribute having the given value as payload IR root. If empty select the pass anchor operation as the payload IR root.
           --debug-transform-root-tag=<string>                   - Select the operation with 'transform.target_tag' attribute having the given value as container IR for top-level transform ops. This allows user control on what transformation to apply. If empty, select the container of the top-level transform op.
           --enable-expensive-checks                             - perform expensive checks to better report errors in the transform IR
           --enforce-single-top-level-transform-op               - Ensure that only a single top-level transform op is present in the IR.
           --test-module-generation                              - test the generation of the transform module during pass initialization, overridden by parsing
           --transform-file-name=<string>                        - Optional filename containing a transform dialect specification to apply. If left empty, the IR is assumed to contain one top-level transform dialect operation somewhere in the module.
           --transform-library-paths=<string>                    - Optional paths to files with modules that should be merged into the transform module to provide the definitions of external named sequences.
         --test-vector-transferop-opt                            -   Test optimization transformations for transfer ops
         --test-vector-warp-distribute                           -   Test vector warp distribute transformation and lowering patterns
         --transform-dialect-check-uses                          -   warn about potential use-after-free in the transform dialect
         --transform-infer-effects                               -   infer transform side effects for symbols
         --transform-interpreter                                 -   transform dialect interpreter
           --debug-payload-root-tag=<string>                     - Select the operation with 'transform.target_tag' attribute having the given value as payload IR root. If empty select the pass anchor operation as the payload IR root.
         --transform-preload-library                             -   preload transform dialect library
           --transform-library-paths=<string>                    - Optional paths to files with modules that should be merged into the transform module to provide the definitions of external named sequences.
     --verify-each                                               - Run the verifier after each transformation pass
     --test-loop-fusion-transformation                           - Enable testing of loop fusion transformation
   ```

Looks like linalg level tiling is hiding inside the transform dialect passes!!

### linalg-tiling?

Files of interest:

- [llvm-project-pistachio/mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h](../mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h)
- [llvm-project-pistachio/mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp](../mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp)
- [llvm-project-pistachio/mlir/test/lib/Dialect/Linalg/TestLinalgTransforms.cpp](../mlir/test/lib/Dialect/Linalg/TestLinalgTransforms.cpp)

```
mlir-opt --linalg-tiling matmul104x104.mlir
```

Let's try regular:

```
mlir-opt matmul104x104.mlir
```

Now let's try with a pass:

```
mlir-opt --test-loop-fusion-transformation  matmul104x104.mlir
```

Or another pass?

```
mlir-opt --test-linalg-transform-patterns matmul104x104.mlir
```



### Affine tiling?

```
cd emily-notes

sh tile-matmul104x104.sh
```

```
$ for lib in  build/lib/* ; do nm $lib 2>/dev/null  | grep printNewline && echo "Found in $lib" ; done

```



```mlir
#affine_map42 = affine_map<(d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)>

// Use an affine mapping definition in an alloc operation, binding the
// SSA value %N to the symbol s0.
%a = memref.alloc()[%N] : memref<4x4xf32, #affine_map42>
```

### Scf tiling?

## Examples

1. Run regular matrix multiplication
   ```
   sh linalg-run-w-mlir-cpu-runner.sh matmul104x104.mlir main out
   ```

   

2. 

## Troubleshooting

Error:

```
JIT session error: Symbols not found: [ _mlir_memref_to_llvm_alloc ]
Error: Failed to materialize symbols: { (main, { main, _mlir_main }) }
```

Solution?

```
for lib in  build-riscv/lib/* ; do nm $lib 2>/dev/null  | grep _mlir_memref_to_llvm_alloc && echo "Found in $lib" ; done
```

Actual solution: remove `use-generic-functions` from the  `--finalize-memref-to-llvm='use-generic-functions index-bitwidth=32'` pass, OR link in your own C code definition of this function.
