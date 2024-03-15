# Getting Started w/ MLIR

Before starting, make sure to add your MLIR LLVM build to your path.

For example,

 `export PATH=/home/hoppip/llvm-project-pistachio/build-riscv/bin:$PATH`

## MLIR + C to LLVM

Automate process with

```
sh mlir-to-llvm.sh <input.mlir> <c-file-in-which-to-embed-mlir.c>
```

For example,

`sh mlir-to-llvm.sh hello.mlir print-fake-tensors.c`

**This `mlir-to-llvm` script will add a main function that prints out the tensor returned by function `simp`,** where `simp` is a function defined in MLIR with signature `func.func @simp(%arg : tensor<2xf32>) -> tensor<2xf32>`.

## C to LLVM

Automate process with

```
sh c-to-llvm.sh simple-c.c
```

(replace `simple-c.c` with your desired input mlir file)

#### [simple-c.c](simple-c.c)

Compile to Executeable: `clang simple-c.c -o out/simple-c.o`

Run executeable: `out/simple-c.o`

C to LLVM Bitcode: `../../build-riscv/bin/clang -O1 -emit-llvm simple-c.c -c -o out/simple-c.bc`

LLVM Bitcode to LLVM IR: `../../build-riscv/bin/llvm-dis < out/simple-c.bc > out/simple-c.ll`

LLVM IR to Exxecuteable: `../../build-riscv/bin/clang out/simple-c.ll -o out/simple-c.o`

## MLIR to LLVM

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

   

   Translate LLVM MLIR dialect to LLVM IR:

   ```
   mlir-translate --mlir-to-llvmir test-in-llvm-dialect.mlir > test.ll
   ```

   Convert .ll file to executable using clang:

   

