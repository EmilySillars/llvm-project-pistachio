# PAAM Resources: Compile and Run simple-c-.c
[[Return to Homepage]](../../paam-resources/README.md)

1. Start in the top level directory of your cloned LLVM repo

2. Build LLVM, after which

- All the LLVM tools built will be located inside the `bin` directory of your build folder
- You can invoke the clang compiler by calling it's corresponding executable file located inside the `bin` directory of your build folder.

3. Suppose the name of your build folder is "build-riscv" and it's located in the top level of your cloned repo.

4. Navigate to the `llvm-simple-c directory`:
   ```
   cd paam-examples/llvm-simple-c
   ```

## Compile and Run

Compile to Executable:

```
../../build-riscv/bin/clang simple-c.c -o out/simple-c.o
```

Run executable:

```
out/simple-c.o
```

## Inspect LLVM IR Representation: clang + llvm-dis

Convert C to LLVM Bitcode:

```
../../build-riscv/bin/clang -O1 -emit-llvm simple-c.c -c -o out/simple-c.bc
```

- Note that `-O1` specifies the optimization level (`-O1` is lowest, `-O3` is highest)


Convert LLVM Bitcode to LLVM IR:

```
../../build-riscv/bin/llvm-dis < out/simple-c.bc > out/simple-c.ll
```

**Compile LLVM IR to an executable:**

```
../../build-riscv/bin/clang out/simple-c.ll -o out/simple-c.o
```