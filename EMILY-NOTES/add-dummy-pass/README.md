# Avocado: add a "Hello World" Pass to mlir-opt

This hello world pass will count all functions and instructions in the input program.

## Reference Files

For adding a new pass, look at

- mlir/include/mlir/Dialect/Linalg/Passes.h
- mlir/include/mlir/Dialect/Linalg/Passes.td
- mlir/lib/Dialect/Linalg/Transforms/ElementwiseToLinalg.cpp

## Set up notes

When you first enter repo, make sure to

1. ```
   export PATH=/home/hoppip/llvm-project-pistachio/build-riscv/bin:$PATH
   ```

2. ```
   export MLIR_CPU_RUNNER_LIBS=/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so
   ```

When you add any new source files, make sure to re-run cmake before re building!

```
cd llvm-project-pistachio

cd build-riscv

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

When you make changes to the MLIR passes and want to test your changes, rebuild!

```
cd llvm-project-pistachio

cd build-riscv

ninja -j 20
```

## Run matmul with mlir-cpu-runner

```
sh run-w-mlir-cpu-runner.sh matmul104x104 main
```

## Run matmul through avocado mlir-opt pass

```
sh run-thru-avocado.sh matmul104x104
```

## Modifications needed to add the pass

Consult [relevant-avocado-changes.diff](relevant-avocado-changes.diff)

## Old notes (delete later!)

- Example input's matmul from here: https://github.com/EmilySillars/iree-fork/blob/tiling/iree-fork/matmul104x104.mlir

- Let's generate same input for our matmul as our iree-cpu example!

- https://github.com/llvm/llvm-project/blob/1a4dd8d36206352220eb3306c3bdea79b6eeffc3/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir

- For running/testing that new pass

  - https://github.com/EmilySillars/Quidditch-zigzag/blob/tiling/runtime/tests/compile-for-riscv.sh
  - EMILY-NOTES/learning-mlir/run-func-memrefs.sh

  What files did I change when adding a new pass?
