# Emily's Notes on Learning LLVM

- Computer Setup instructions [here](https://github.com/EmilySillars/llvm-project-pistachio/blob/learn-llvm/llvm-mlir-riscv-setup.md)

- Official Guide for Adding a Compiler Pass [here](https://llvm.org/docs/WritingAnLLVMNewPMPass.html)

## Adding a Pass to LLVM

1) put `passName.h` in `llvm/include/llvm/Tranforms/Utils`
2) put `passName.cpp` in `llvm/lib/Transforms/Utils`
3) update `llvm/lib/Transforms/Utils/CMakeLists.txt` with `passName.cpp`
4) update `llvm/lib/Passes/PassBuilder.cpp` with the line `#include "llvm/Transforms/Utils/passName.h"`
5) update `llvm/lib/Passes/PassRegistry.def` with the line `FUNCTION_PASS("passName", PassNamePass())`

## Invoking Pistachio Pass (Emily's custom pass)

Execute these instructions from inside EMILY-NOTES directory...
```
../build-riscv/bin/clang tests/hello.c                                      # compile to executeable
../build-riscv/bin/clang -O3 -S -emit-llvm tests/hello.c -c -o out/hello.ll # compile to human readable LLVM IR
../build-riscv/bin/clang -O3 -emit-llvm tests/hello.c -c -o out/hello.bc    # compile to LLVM bitcode
../build-riscv/bin/opt -disable-output out/hello.bc -passes=pistachio       # run pistachio pass over the LLVM bitcode
```

