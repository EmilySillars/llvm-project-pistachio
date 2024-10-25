# lowering matmul to LLVM IR

### 1) compile to llvm bitcode

```
<LLVM-BUILD-DIRECTORY>/bin/clang -O1 -emit-llvm -c matmul-in-llvm.c
```

### 2) disassemble the llvm bitcode into human-readable LLVM IR

```
<LLVM-BUILD-DIRECTORY>/bin/llvm-dis matmul-in-llvm.bc
```

   - flag `-emit-llvm` instructs clang to emit llvm ir bitcode
   - flag `O1` disables optimization passes
   - flag `-c` prevents linking (no LLVM IR generated for linked libraries)