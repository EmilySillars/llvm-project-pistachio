# Avocado: add a "Hello World" Pass to mlir-opt

This hello world pass will count all the instructions in the input program, and print out their names.

## Reference Files

For adding a new pass

- mlir/include/mlir/Dialect/Linalg/Passes.h
- mlir/include/mlir/Dialect/Linalg/Passes.td
- mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp

For running/testing that new pass

- https://github.com/EmilySillars/Quidditch-zigzag/blob/tiling/runtime/tests/compile-for-riscv.sh
- EMILY-NOTES/learning-mlir/run-func-memrefs.sh

## Set up notes

When you first enter repo, make sure to

1. ```
   export PATH=/home/hoppip/llvm-project-pistachio/build-riscv/bin:$PATH
   ```

2. ```
   export MLIR_CPU_RUNNER_LIBS=/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so
   ```

When you make changes to the MLIR passes and want to test your changes, rebuild!

```
cd llvm-project-pistachio

cd build-riscv

ninja -j 20
```

## Testing the pass

- Example input's matmul from here: https://github.com/EmilySillars/iree-fork/blob/tiling/iree-fork/matmul104x104.mlir
- Let's generate same input for our matmul as our iree-cpu example!
- https://github.com/llvm/llvm-project/blob/1a4dd8d36206352220eb3306c3bdea79b6eeffc3/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir
- 

