# Avocado: add a "Hello World" Pass to mlir-opt

This hello world pass will count all the instructions in the input program, and print out their names.

## Reference Files

For adding a new pass

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

## Testing the pass

```
sh run-thru-avocado.sh matmul104x104
```

## Modifications needed to add the pass

```
diff --git a/mlir/include/mlir/Dialect/Linalg/Passes.h b/mlir/include/mlir/Dialect/Linalg/Passes.h
index 5f46affe592a..fb10ab5b43ed 100644
--- a/mlir/include/mlir/Dialect/Linalg/Passes.h
+++ b/mlir/include/mlir/Dialect/Linalg/Passes.h
@@ -29,6 +29,9 @@ struct OneShotBufferizationOptions;
 #define GEN_PASS_DECL
 #include "mlir/Dialect/Linalg/Passes.h.inc"
 
+/// Create a "hello world" pass at the linalg level
+std::unique_ptr<Pass> createAvocadoPass();
+
 std::unique_ptr<Pass> createConvertElementwiseToLinalgPass();
 
 std::unique_ptr<Pass> createLinalgFoldUnitExtentDimsPass();
diff --git a/mlir/include/mlir/Dialect/Linalg/Passes.td b/mlir/include/mlir/Dialect/Linalg/Passes.td
index cca50e21d5ce..7840c831377b 100644
--- a/mlir/include/mlir/Dialect/Linalg/Passes.td
+++ b/mlir/include/mlir/Dialect/Linalg/Passes.td
@@ -11,6 +11,16 @@
 
 include "mlir/Pass/PassBase.td"
 
+def Avocado : Pass<"avocado", ""> {
+  let summary = "A 'hello world' pass at the linalg level";
+  let description = [{
+    Counts all instructions and prints function names.
+  }];
+  let constructor = "mlir::createAvocadoPass()";
+  let dependentDialects = ["linalg::LinalgDialect"];
+}
+
+
 def ConvertElementwiseToLinalg : Pass<"convert-elementwise-to-linalg", ""> {
   let summary = "Convert ElementwiseMappable ops to linalg";
   let description = [{
diff --git a/mlir/lib/Dialect/Linalg/Transforms/CMakeLists.txt b/mlir/lib/Dialect/Linalg/Transforms/CMakeLists.txt
index 4f47e3b87184..f8a7010b67a4 100644
--- a/mlir/lib/Dialect/Linalg/Transforms/CMakeLists.txt
+++ b/mlir/lib/Dialect/Linalg/Transforms/CMakeLists.txt
@@ -11,6 +11,7 @@ add_mlir_dialect_library(MLIRLinalgTransforms
   DropUnitDims.cpp
   ElementwiseOpFusion.cpp
   ElementwiseToLinalg.cpp
+  Avocado.cpp
   EliminateEmptyTensors.cpp
   EraseUnusedOperandsAndResults.cpp
   FusePadOpWithLinalgProducer.cpp

```

## Old notes

- Example input's matmul from here: https://github.com/EmilySillars/iree-fork/blob/tiling/iree-fork/matmul104x104.mlir

- Let's generate same input for our matmul as our iree-cpu example!

- https://github.com/llvm/llvm-project/blob/1a4dd8d36206352220eb3306c3bdea79b6eeffc3/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir

- For running/testing that new pass

  - https://github.com/EmilySillars/Quidditch-zigzag/blob/tiling/runtime/tests/compile-for-riscv.sh
  - EMILY-NOTES/learning-mlir/run-func-memrefs.sh

  What files did I change when adding a new pass?
