## Build LLVM with MLIR and RISC-V target

*Based on Ravi's notes and [https://mlir.llvm.org/getting_started/](https://urldefense.com/v3/__https://mlir.llvm.org/getting_started/__;!!D9dNQwwGXtA!SEOGnQ6kNZqQqZPcpFmf8I3xb2_IWywnYuC-onD9gFeQlo2vOJYwE8HHNPkxP2PYzkO6OKhJlpcbI77vcrKwuA$)*

1. Clone LLVM (which we already did) and then make a build directory and go into it:

```
cd llvm-project-pistachio; \
mkdir build-riscv; \
cd build-riscv
```

2. Prepare makefiles using following options:

If you don't need mlir, replace `-DLLVM_ENABLE_PROJECTS="mlir;clang;lld"` with `-DLLVM_ENABLE_PROJECTS="clang;lld"`

```
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

then build with:

```
ninja -j 20
```
