# Getting Started w/ MLIR

Before starting, make sure to add your MLIR LLVM build to your path.

For example,

 `export PATH=/home/hoppip/llvm-project-pistachio/build-riscv/bin:$PATH`

In the same vein, make sure to set the env variable `MLIR-CPU-RUNNER-LIBS` to the corresponding `.so` files in your build.

For example,

```
export MLIR_CPU_RUNNER_LIBS=/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so
```

## Quick Examples w/ mlir-cpu-runner

- [Print out a tensor](print-tensors.mlir)

  ```
  sh run-func-as-mlir.sh print-tensors.mlir main
  ```

- [Loop through a vector](practice-scf.mlir)

  ```
  sh run-func-as-mlir.sh practice-scf.mlir main
  ```

- [Print out a matrix](print-memrefs.mlir) (represented as a memref)

  ```
  sh run-func-memrefs.sh print-memrefs.mlir main
  ```

- [Print out even more matrices](print-memrefs2.mlir) (represented as memrefs)

  ```
  sh run-func-memrefs.sh print-memrefs2.mlir main
  ```

- [Multiply two matrices](print-memrefs-qmat.mlir) (quantized matrix multiplication)

  ```
  sh run-func-memrefs.sh print-memrefs-qmat.mlir main
  ```

- [Print out *part* of a matrix](memref-subview-ex.mlir) (represented as a memref.subview)

  ```
  sh run-func-as-mlir.sh memref-subview-ex.mlir main
  ```

- [Split a matrix into tiles](print-memrefs-tiled.mlir) (using memref.subview) 

  ```
  sh run-func-memrefs.sh print-memrefs-tiled.mlir main
  ```

- [Tiled Matrix Multiplication](print-memrefs-qmat-tiled.mlir) (QMAT) 
  
  ```
  sh run-func-memrefs.sh print-memrefs-qmat-tiled.mlir main
  ```
  
- what next?



## Printing MLIR tensor with JIT (mlir-cpu-runner)

- `printfMemrefF32` prints out a tensor!

- You can create a main function using the `func` dialect.
- Markus suggests minimal example [here](https://github.com/llvm/llvm-project/blob/1a4dd8d36206352220eb3306c3bdea79b6eeffc3/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir)

Running this minimal example (use `mlir-cpu-runner`! [mlir-cpu-runner is a JIT compiler :/](https://mlir.llvm.org/getting_started/TestingGuide/#integration-tests) )

- `%s` is the input file path, for ex `/home/hoppip/llvm-project-pistachio/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir`

- `%mlir_c_runner_utils` and `%mlir_runner_utils` are respective paths to `.so` files like `libmlir_c_runner_utils.so`
  for ex, `/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so`

  and `/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so`

```
// RUN: mlir-opt %s -test-linalg-transform-patterns=test-linalg-to-vector-patterns \
// RUN: -empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize \
// RUN: -bufferization-bufferize -tensor-bufferize -func-bufferize \
// RUN: -finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref \
// RUN: -convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata \
// RUN: -lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils \
// RUN: | FileCheck %s
```

***Method 1:***

Path-specific steps as follows:

```
rm hoodle.mlir; \
mlir-opt /home/hoppip/llvm-project-pistachio/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir \
-test-linalg-transform-patterns=test-linalg-to-vector-patterns \
-empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize \
-bufferization-bufferize -tensor-bufferize -func-bufferize \
-finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref \
-convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata \
-lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts > hoodle.mlir &&
mlir-cpu-runner -e main -entry-point-result=void \
-shared-libs=/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_c_runner_utils.so,/home/hoppip/llvm-project-pistachio/build-riscv/lib/libmlir_runner_utils.so hoodle.mlir

```

***Method 2:*** **(not working)**

If you don't want to use the [mlir-cpu-runner](https://mlir.llvm.org/getting_started/TestingGuide/#integration-tests), you can do

```
rm hoodle.mlir; clear; \
mlir-opt /home/hoppip/llvm-project-pistachio/mlir/test/Integration/Dialect/Linalg/CPU/test-padtensor.mlir \
-test-linalg-transform-patterns=test-linalg-to-vector-patterns \
-empty-tensor-to-alloc-tensor -linalg-bufferize -arith-bufferize \
-bufferization-bufferize -tensor-bufferize -func-bufferize \
-finalizing-bufferize -buffer-deallocation-pipeline -convert-bufferization-to-memref \
-convert-linalg-to-loops -convert-scf-to-cf -expand-strided-metadata \
-lower-affine -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts > out/hoodle.mlir && \
mlir-translate --mlir-to-llvmir out/hoodle.mlir > out/hoodle.ll &&
clang out/hoodle.ll -o out/hoodle.o
```

Error:

```
warning: overriding the module target triple with x86_64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
ld: error: undefined symbol: memrefCopy
>>> referenced by LLVMDialectModule
>>>               /tmp/hoodle-c44ca6.o:(main)

ld: error: undefined symbol: printMemrefF32
>>> referenced by LLVMDialectModule
>>>               /tmp/hoodle-c44ca6.o:(main)
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

Possible solution: https://discourse.llvm.org/t/symbols-not-found-memrefcopy/4580

Or maybe I need to change my C file to a CPP file and manually include [/home/hoppip/llvm-project-pistachio/mlir/include/mlir/ExecutionEngine/RunnerUtils.h](/home/hoppip/llvm-project-pistachio/mlir/include/mlir/ExecutionEngine/RunnerUtils.h) ???

## Printing MLIR tensor with a C function

Automate process with

```
sh mlir-to-llvm.sh <input.mlir> <c-file-in-which-to-embed-mlir.c>
```

For example,

`sh mlir-to-llvm.sh hello.mlir print-fake-tensors.c`

**This `mlir-to-llvm` script will add a main function that prints out the tensor returned by function `simp`,** where `simp` is a function defined in MLIR with signature `func.func @simp(%arg : tensor<2xf32>) -> tensor<2xf32>`.

## Compile C to LLVM + Run

Automate process with

```
sh c-to-llvm.sh <input.c>
```

For example,

```
sh c-to-llvm.sh simple-c.c
```

1. ### [simple-c.c](simple-c.c)

Compile to Executeable: `clang simple-c.c -o out/simple-c.o`

Run executeable: `out/simple-c.o`

C to LLVM Bitcode: `../../build-riscv/bin/clang -O1 -emit-llvm simple-c.c -c -o out/simple-c.bc`

LLVM Bitcode to LLVM IR: `../../build-riscv/bin/llvm-dis < out/simple-c.bc > out/simple-c.ll`

LLVM IR to Exxecuteable: `../../build-riscv/bin/clang out/simple-c.ll -o out/simple-c.o`

## Compile MLIR to LLVM + Run

Automate process with 

```
sh to-llvm.sh minimal.mlir
```

(replace `minimal.mlir` with your desired input mlir file)

1) #### [minimal.mlir](minimal.mlir)

   Convert all dialects to LLVM dialect:

   ```
   mlir-opt minimal.mlir  --one-shot-bufferize='bufferize-function-boundaries' -test-lower-to-llvm > out/minimal-in-llvm-dialect.mlir
   ```

   Translate LLVM MLIR dialect to LLVM IR:

   ```
   mlir-translate --mlir-to-llvmir out/minimal-in-llvm-dialect.mlir > out/minimal.ll
   ```

   Convert .ll file to executable using clang:

   *Remember to add a main function to the LLVM IR / .ll file!!! (see [frankenstein.ll](frankenstein.ll))*

   ```
   clang out/minimal.ll -o out/minimal.o
   ```

   

2) #### [test.mlir](test.mlir)

   ##### *Method 1:*

   Convert all dialects to LLVM dialect:

   ```
   mlir-opt test.mlir  --one-shot-bufferize='bufferize-function-boundaries' -test-lower-to-llvm > out/test-in-llvm-dialect.mlir
   ```

   Translate LLVM MLIR dialect to LLVM IR:

   ```
   mlir-translate --mlir-to-llvmir out/test-in-llvm-dialect.mlir > out/test.ll
   ```

   Convert .ll file to executable using clang:

   *Remember to add a main function to the LLVM IR / .ll file!!! (see [frankenstein.ll](frankenstein.ll))*

   ```
   clang out/test.ll -o out/test.o
   ```

   ##### *Method 2:* (Not working)

   Convert all dialects to LLVM dialect:

   ```
   mlir-opt test.mlir --convert-scf-to-cf --convert-cf-to-llvm --func-bufferize --convert-func-to-llvm          --convert-arith-to-llvm --expand-strided-metadata --normalize-memrefs          --memref-expand --fold-memref-alias-ops --finalize-memref-to-llvm --reconcile-unrealized-casts > test-in-llvm-dialect.mlir
   ```
   
   Error:

   ```
   test.mlir:6:7: error: failed to legalize operation 'builtin.unrealized_conversion_cast' that was explicitly marked illegal
         %lhs : tensor<10xf32>,
         ^
   test.mlir:6:7: note: see current operation: %6 = "builtin.unrealized_conversion_cast"(%5) : (!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>) -> memref<10xf32>
   ```
   
   

### SCF notes

```
    //  %result1, %result2 = scf.for %i = %c0 to %c8 step %c1 iter_args(%partial = %init, %hoodle = %x) -> (f32, index) {
    //     // extract subview (tile)
    //     %tile= memref.subview %const[%x,%y][16,2][1,1]  : memref<16x16xf32> to memref<16x2xf32, strided<[16, 1], offset: ?>>
    //     // print the tile
    //     // %cast3 = memref.cast %tile : memref<16x2xf32, strided<[16, 1], offset: ?>> to memref<*xf32>
    //     // func.call @printMemrefF32(%cast3) : (memref<*xf32>) -> ()
    //     // update indices
    //     func.call @myPrintF32(%partial) : (f32) -> ()
    //     // %x1 = arith.addf %x, %i : index
    //     // just incrementing a float each iter
    //     %one = arith.constant 1.0 : f32
    //     %updated = arith.addf %partial, %one : f32
    //     %updated2 = arith.addi %hoodle, %c0 : index
    //     scf.yield %updated, %updated2 : f32,index 
    //     //scf.yield %updated2 : index
    //  }
```

more notes

```
  //   %tile_left = memref.subview %arg0[%zero,%i][16,2][1,1]  : memref<16x16xi8> to memref<16x2xi8, strided<[16, 1], offset: ?>>
 //     %tile_right = memref.subview %arg1[%i,%zero][2,16][1,1]  : memref<16x16xi8, strided<[1, 16]>> to memref<2x16xi8, strided<[16, 1], offset: ?>>
   //   %tl_unstrided = memref.cast %tile_left : memref<16x2xi8, strided<[16, 1], offset: ?>> to memref<16x2xi8>
     // %tr_unstrided = memref.cast %tile_left : memref<16x2xi8, strided<[16, 1], offset: ?>> to memref<16x2xi8, strided<[16,1]>>
```

